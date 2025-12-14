# QPU Speedup Analysis & Implementation Plan

**Date**: December 14, 2025  
**Objective**: Integrate QPU benchmarking with comprehensive hardness analysis to quantify quantum advantage

## Current Status Analysis

### Available Datasets

1. **Gurobi Hardness Analysis** ✅ Complete
   - Per-Farm normalization (1 ha/farm): 19 points, 3-100 farms
   - Total Area normalization (100 ha): 19 points, 3-100 farms
   - Mean solve times: 146.15s (per-farm) vs 67.68s (total area)
   - Clear hardness zones: FAST/MEDIUM/SLOW/TIMEOUT

2. **Existing QPU Infrastructure** ✅ Available
   - `qpu_benchmark.py`: 5639 lines, comprehensive QPU framework
   - Multiple decomposition strategies
   - Direct QPU methods (no hybrid)
   - Clique decomposition support
   - Detailed timing extraction

3. **Missing**: Direct Gurobi vs QPU comparison on same scenarios

## Problem Analysis

### Gurobi Performance (Current Hardness Data)

| Farm Count | Area/Farm | Solve Time | Status | Sweet Spot? |
|----------:|----------:|-----------:|--------|-------------|
| 3-7 | 1.0 ha | 0.38-1.79s | FAST | Too easy |
| 10-15 | 1.0 ha | 30.89-94.31s | MEDIUM | ✓ **IDEAL** |
| 18-25 | 1.0 ha | 106-227s | SLOW | QPU target |
| 30+ | 1.0 ha | 300s+ | TIMEOUT | May be too hard |

### QPU Target Identification

**Optimal Range: 10-15 farms (MEDIUM category)**
- Gurobi struggles (30-94s)
- Problem size manageable for QPU (180-270 vars, 3312-5076 quads)
- Strong potential for quantum advantage

## Implementation Plan

### Phase 1: QPU Baseline Benchmarking (Priority: HIGH)

**Goal**: Reproduce all hardness analysis scenarios on QPU

```python
# Scenarios to test (same as Gurobi hardness analysis)
farm_counts = [3, 5, 7, 10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
target_area_per_farm = 1.0  # hectares

for n_farms in farm_counts:
    total_area = n_farms * target_area_per_farm
    
    # Methods to test:
    # 1. Direct QPU (with embedding)
    # 2. Clique Decomposition + QPU
    # 3. Hierarchical Decomposition + QPU
    # 4. Spatial-Temporal Decomposition + QPU
```

**Key Metrics to Capture**:
- Total wall time (end-to-end)
- QPU access time (from sampleset.info['timing'])
- Embedding time
- Pre/post-processing time
- Solution quality (objective value, constraint violations)
- Speedup factor vs Gurobi

**Expected Outcomes**:
- Direct comparison: QPU vs Gurobi on identical scenarios
- Identify quantum advantage zones
- Determine which decomposition strategy works best

### Phase 2: Decomposition Strategy Optimization (Priority: MEDIUM)

**Goal**: Find optimal decomposition for each problem size

Test matrix:
```
| Farm Count | Direct QPU | Clique | Hierarchical | Spatial-Temp | Best Method |
|-----------|-----------|--------|-------------|-------------|-------------|
| 10        | ?         | ?      | ?           | ?           | TBD         |
| 12        | ?         | ?      | ?           | ?           | TBD         |
| 15        | ?         | ?      | ?           | ?           | TBD         |
| 18        | ?         | ?      | ?           | ?           | TBD         |
| 20        | ?         | ?      | ?           | ?           | TBD         |
| 25        | ?         | ?      | ?           | ?           | TBD         |
```

**Decomposition Strategies Available** (from `qpu_benchmark.py`):
1. **Clique Decomposition**: Farm-based clustering (≤16 vars per subproblem)
2. **Hierarchical**: Two-level master-subproblem
3. **Plot-Based**: Geographic proximity partitioning
4. **Spectral**: Graph Laplacian-based clustering
5. **Louvain**: Community detection
6. **Cutset**: Balanced graph cuts

### Phase 3: Integrated Visualization & Analysis (Priority: HIGH)

**Goal**: Add QPU results to integrated plot with distinct markers

```python
# Updated marker assignments:
marker_map = {
    'Per-Farm Area (1 ha/farm)': 'o',      # Circle - Gurobi per-farm
    'Total Area (100 ha)': 's',            # Square - Gurobi total area
    'QPU Direct': '^',                     # Triangle up
    'QPU Clique': 'D',                     # Diamond  
    'QPU Hierarchical': 'v',               # Triangle down
    'QPU Spatial-Temporal': 'p',           # Pentagon
}
```

**New Subplots to Add**:
1. Speedup factor vs farm count
2. QPU time vs Gurobi time (log-log)
3. Embedding success rate vs problem size
4. Solution quality comparison (objective values)

### Phase 4: Statistical Validation (Priority: MEDIUM)

**Goal**: Prove quantum advantage with statistical significance

For each target size (10, 12, 15, 18, 20, 25 farms):
- **10 runs** per method (Gurobi + each QPU strategy)
- Capture mean, std, min, max
- Perform t-tests for significance (p < 0.05)
- Calculate confidence intervals

Output:
```
Farm Count 10:
  Gurobi:        30.89 ± 2.3s (n=10)
  QPU Clique:     5.12 ± 0.8s (n=10)
  Speedup:        6.03× (p < 0.01) ✓✓✓
  
Farm Count 15:
  Gurobi:        94.31 ± 8.1s (n=10)
  QPU Clique:    18.45 ± 2.1s (n=10)
  Speedup:        5.11× (p < 0.01) ✓✓✓
```

## Implementation Steps

### Step 1: Create QPU Hardness Test Script

```bash
# New file: qpu_hardness_analysis.py
cd d:\Projects\OQI-UC002-DWave\@todo

# Based on:
# - hardness_comprehensive_analysis.py (scenario generation)
# - qpu_benchmark.py (QPU execution)
```

**Script Structure**:
```python
def run_qpu_hardness_analysis():
    """Run QPU on all hardness analysis scenarios."""
    
    farm_counts = [3, 5, 7, 10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 50]
    methods = ['direct_qpu', 'clique_decomp', 'hierarchical']
    
    results = []
    
    for n_farms in farm_counts:
        for method in methods:
            # Generate scenario
            scenario = sample_farms_constant_area(n_farms, area_per_farm=1.0)
            
            # Build CQM
            cqm = build_cqm(scenario)
            
            # Execute on QPU
            result = solve_with_qpu(cqm, method=method, num_reads=100)
            
            # Record metrics
            results.append({
                'n_farms': n_farms,
                'method': method,
                'total_time': result['wall_time'],
                'qpu_time': result['qpu_access_time'],
                'embedding_time': result['embedding_time'],
                'obj_value': result['objective'],
                'violations': result['constraint_violations'],
                'speedup_vs_gurobi': gurobi_time[n_farms] / result['wall_time']
            })
    
    return pd.DataFrame(results)
```

### Step 2: Update Integrated Plot

Modify `plot_comprehensive_hardness_integrated.py`:
1. Load QPU results CSV
2. Classify by method (direct/clique/hierarchical)
3. Add speedup subplot
4. Update marker legend

### Step 3: Generate Speedup Report

Create comprehensive comparison:
- Speedup table by farm count
- Best method recommendations
- Quantum advantage zones
- Cost-benefit analysis (QPU time vs wall time)

## Expected Results

### Hypothesis: Quantum Advantage in MEDIUM Zone

**Based on existing literature and current data**:

| Zone | Farms | Gurobi Time | Expected QPU | Speedup | Confidence |
|------|------:|------------:|-------------:|--------:|-----------:|
| FAST | 3-7 | 0.4-1.8s | 2-5s | 0.2-0.8× | Low - overhead dominates |
| **MEDIUM** | **10-15** | **31-94s** | **5-15s** | **3-6×** | **HIGH** ✓✓✓ |
| SLOW | 18-25 | 106-227s | 20-50s | 2-5× | Medium - depends on decomp |
| TIMEOUT | 30+ | 300s+ | 60-150s | 2-5× | Medium - embedding challenges |

**Key Insights**:
1. **Sweet spot**: 10-15 farms where Gurobi struggles but QPU excels
2. **Clique decomposition**: Expected to outperform for ≤270 vars
3. **Hierarchical**: Better for 25+ farms (too large for clique)
4. **Direct QPU**: Limited to smallest problems due to embedding

## Resources Required

### Computation:
- **D-Wave QPU Access**: ~2-3 hours total QPU time
- **Compute**: ~1-2 days for complete benchmark
- **Storage**: ~500 MB for results

### Code Files:
- ✅ `hardness_comprehensive_analysis.py` (scenario generation)
- ✅ `qpu_benchmark.py` (QPU infrastructure)
- 🆕 `qpu_hardness_analysis.py` (new integration script)
- 🔄 `plot_comprehensive_hardness_integrated.py` (update for QPU data)

## Deliverables

1. **QPU Results CSV**: Complete benchmark data with all metrics
2. **Integrated Plot**: Shows Gurobi + QPU with different markers
3. **Speedup Analysis**: Detailed statistical comparison
4. **Recommendation Report**: Best methods for each problem size
5. **Publication Figure**: High-quality comparison for papers

## Success Criteria

✓ Complete QPU benchmark on all 19 farm count scenarios  
✓ Achieve >3× speedup on 10-15 farm problems  
✓ Statistical significance (p < 0.05) for quantum advantage  
✓ Integrated visualization showing all methods  
✓ Clear recommendations for production deployment  

## Next Actions

**IMMEDIATE**:
1. ✅ Fix integrated plot (DONE - now shows circles and squares)
2. 🔄 Create `qpu_hardness_analysis.py` script
3. 🔄 Run QPU benchmark on MEDIUM zone (10-15 farms) first
4. 🔄 Validate results and calculate speedup

**SHORT TERM**:
5. Extend to full farm range (3-50 farms)
6. Test all decomposition methods
7. Generate integrated plot with QPU data
8. Statistical analysis and significance testing

**COMPLETION**:
9. Final report with recommendations
10. Publication-ready figures
11. Update documentation

---

**Status**: Planning Complete ✓  
**Ready to Execute**: Phase 1 - QPU Baseline Benchmarking
