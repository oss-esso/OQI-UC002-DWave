# QPU Benchmark: Small Scenario Testing Guide

## Overview

The QPU benchmark has been enhanced to support small synthetic scenarios designed specifically for direct QPU embedding. These scenarios range from 6 to 160 variables, making them ideal for testing direct quantum annealing without requiring decomposition strategies.

## Available Small Scenarios

| Scenario | Plots | Foods | Total Variables | Description |
|----------|-------|-------|-----------------|-------------|
| `micro_6` | 2 | 2 | 6 | Minimal test case |
| `micro_12` | 3 | 3 | 12 | Very small |
| `tiny_24` | 4 | 5 | 25 | Tiny problem |
| `tiny_40` | 5 | 6 | 36 | Small problem |
| `small_60` | 6 | 8 | 56 | Small-medium |
| `small_80` | 7 | 10 | 80 | Medium |
| `small_100` | 8 | 11 | 99 | Medium-large |
| `medium_120` | 9 | 12 | 120 | Large embeddable |
| `medium_160` | 10 | 14 | 154 | Maximum embeddable |

**Note**: Total variables = (Plots × Foods) + Foods
- Y variables: Plot-food assignments (Plots × Foods)
- U variables: Food selection indicators (Foods)

## Scenario Characteristics

All small scenarios maintain realistic problem structure:
- **Food groups**: Grains, Legumes, Vegetables, Fruits, Proteins (scaled by size)
- **Constraints**: 
  - At most one crop per plot
  - Minimum diversity: at least 1 food per group
  - Food group constraints preserved
- **No rotation/synergy**: Simple formulation for direct embedding testing
- **Realistic benefits**: Scaled nutritional/environmental/economic values

## Usage Examples

### Test a Single Small Scenario

```bash
python qpu_benchmark.py --scenario micro_6
```

This will run the benchmark on the `micro_6` scenario with default methods (ground_truth + direct_qpu).

### Test Multiple Small Scenarios

```bash
python qpu_benchmark.py --scenario micro_6 tiny_24 small_60
```

### Test All Small Scenarios

```bash
python qpu_benchmark.py --all-small
```

This benchmarks all 9 small scenarios from `micro_6` to `medium_160`.

### Specify Custom Methods

```bash
python qpu_benchmark.py --scenario small_60 --methods ground_truth direct_qpu
```

Available methods for small scenarios:
- `ground_truth`: Gurobi optimal solution
- `direct_qpu`: Direct CQM → BQM → QPU embedding (recommended for small scenarios)
- Decomposition methods (optional, but unnecessary for small problems)

### Custom Output File

```bash
python qpu_benchmark.py --all-small --output small_scenarios_benchmark.json
```

## Default Behavior

When using `--scenario` or `--all-small`, the benchmark automatically:

1. **Enables direct QPU embedding**: Small scenarios are designed to embed directly
2. **Uses minimal method set**: Defaults to `ground_truth` + `direct_qpu` only
3. **Skips decomposition**: No need for partitioning at these scales

## Comparison: Traditional vs Small Scenarios

### Traditional Scale-Based Benchmark

```bash
# Uses full_family scenario with 27 foods
python qpu_benchmark.py --scale 25 50 100
```
- Loads standard scenarios with 27 foods per scale
- Total variables: 25×27+27 = 702 variables (for 25 farms)
- Requires decomposition for QPU embedding

### Small Scenario Benchmark

```bash
# Uses synthetic scenarios with controlled sizes
python qpu_benchmark.py --scenario micro_6 tiny_24 small_60
```
- Loads scenarios with 2-8 foods (controlled)
- Total variables: 6, 25, 56 (respectively)
- **Can embed directly on QPU** without decomposition

## Expected Embedding Success

Based on D-Wave Advantage (Pegasus topology, ~5600 qubits):

| Scenario | Variables | Expected Embedding | Chain Length (est.) |
|----------|-----------|-------------------|---------------------|
| micro_6 | 6 | ✅ Trivial | < 1.5 |
| micro_12 | 12 | ✅ Easy | < 2.0 |
| tiny_24 | 25 | ✅ Easy | < 2.5 |
| tiny_40 | 36 | ✅ Easy | < 3.0 |
| small_60 | 56 | ✅ Good | 2-4 |
| small_80 | 80 | ✅ Good | 3-5 |
| small_100 | 99 | ✅ Moderate | 4-6 |
| medium_120 | 120 | ✅ Moderate | 5-8 |
| medium_160 | 154 | ⚠️ Challenging | 6-10 |

**Note**: Actual chain lengths depend on BQM structure (constraint density, coupling strength).

## Output Structure

Results saved to `qpu_benchmark_results/qpu_benchmark_<timestamp>.json`:

```json
{
  "timestamp": "2025-12-09T...",
  "scenarios": ["micro_6", "tiny_24", "small_60"],
  "methods": ["ground_truth", "direct_qpu"],
  "results": [
    {
      "scenario": "micro_6",
      "n_farms": 2,
      "metadata": {
        "n_variables": 6,
        "n_constraints": ...
      },
      "ground_truth": { ... },
      "method_results": {
        "direct_qpu": {
          "objective": ...,
          "qpu_access_time": ...,
          "embedding_time": ...,
          "chain_length": { "mean": ..., "max": ... }
        }
      }
    }
  ]
}
```

## Testing Verification

Verify scenarios load correctly:

```bash
python test_qpu_small_scenarios.py
```

Expected output: ✅ All 9 scenarios load with correct variable counts.

## Use Cases

### 1. **QPU Embedding Testing**
Test if your problem structure embeds efficiently on QPU hardware:
```bash
python qpu_benchmark.py --scenario small_60 --methods direct_qpu
```

### 2. **Quantum vs Classical Comparison**
Compare QPU annealing with Gurobi on embeddable problems:
```bash
python qpu_benchmark.py --all-small --methods ground_truth direct_qpu
```

### 3. **Chain Length Analysis**
Study how chain lengths grow with problem size:
```bash
python qpu_benchmark.py --scenario micro_6 tiny_24 small_60 small_100
# Analyze chain_length.mean in output JSON
```

### 4. **Rapid Prototyping**
Test formulation changes quickly on small problems before scaling up:
```bash
python qpu_benchmark.py --scenario micro_6  # ~10 seconds
```

## Advanced: Mixing Scenarios and Scales

You cannot mix `--scenario` with `--scale` in one run. Choose one approach:

**❌ Invalid:**
```bash
python qpu_benchmark.py --scenario micro_6 --scale 25
```

**✅ Valid (scenarios):**
```bash
python qpu_benchmark.py --scenario micro_6 tiny_24
```

**✅ Valid (scales):**
```bash
python qpu_benchmark.py --scale 25 50
```

## Troubleshooting

### Scenario not found
```
Warning: Unknown scenarios: ['micro_7']
Available scenarios: ['micro_6', 'micro_12', ...]
```
**Solution**: Use exact scenario names from the table above.

### QPU token required
```
Warning: No D-Wave token available. Only ground_truth method will work.
```
**Solution**: Set token via `--token` or environment variable:
```bash
export DWAVE_API_TOKEN="your-token-here"
python qpu_benchmark.py --scenario micro_6
```

### Embedding timeout
If even small scenarios timeout during embedding:
- Check BQM constraint density (may indicate formulation issue)
- Verify QPU connectivity
- Try even smaller scenario (micro_6)

## Next Steps

After validating small scenarios work:

1. **Test incrementally larger scenarios** to find embedding limits
2. **Compare direct_qpu vs decomposition methods** on boundary cases (small_100, medium_120)
3. **Scale up to traditional benchmarks** using decomposition strategies

---

**See also:**
- `qpu_benchmark.py --help` for all options
- Main benchmark documentation for decomposition methods
- D-Wave documentation for embedding best practices
