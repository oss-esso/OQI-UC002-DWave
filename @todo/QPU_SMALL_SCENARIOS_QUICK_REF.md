# QPU Benchmark: Small Scenarios Quick Reference

## Command Cheat Sheet

```bash
# Test single small scenario (6 variables)
python qpu_benchmark.py --scenario micro_6

# Test multiple scenarios
python qpu_benchmark.py --scenario micro_6 tiny_24 small_60

# Test all 9 small scenarios
python qpu_benchmark.py --all-small

# Traditional scale-based (unchanged)
python qpu_benchmark.py --scale 25 50 100

# Custom methods
python qpu_benchmark.py --scenario small_60 --methods ground_truth direct_qpu

# Save to specific file
python qpu_benchmark.py --all-small --output my_results.json
```

## Scenario Quick Reference

| Name | Vars | Plots | Foods | Use Case |
|------|------|-------|-------|----------|
| `micro_6` | 6 | 2 | 2 | Minimal test / debugging |
| `micro_12` | 12 | 3 | 3 | Very small prototype |
| `tiny_24` | 25 | 4 | 5 | Small test case |
| `tiny_40` | 36 | 5 | 6 | Medium-small |
| `small_60` | 56 | 6 | 8 | Good balance for testing |
| `small_80` | 80 | 7 | 10 | Medium complexity |
| `small_100` | 99 | 8 | 11 | Near embedding limit |
| `medium_120` | 120 | 9 | 12 | Challenging embedding |
| `medium_160` | 154 | 10 | 14 | Maximum embeddable |

## What Changed

**Before**: Only large scale tests (25+ farms, 675+ variables), requiring decomposition

**After**: Can test tiny problems (6-154 variables) with direct QPU embedding

## Validation

```bash
# Verify scenarios load correctly
python test_qpu_small_scenarios.py

# Expected: ✅ All small scenarios loaded successfully!
```

## When to Use What

### Use Small Scenarios (`--scenario`) when:
- ✅ Testing direct QPU embedding
- ✅ Rapid prototyping / debugging
- ✅ Learning quantum annealing
- ✅ Validating formulation changes
- ✅ Comparing QPU vs classical on embeddable problems

### Use Traditional Scales (`--scale`) when:
- ✅ Benchmarking decomposition methods
- ✅ Large-scale testing
- ✅ Production scenarios
- ✅ Testing scalability

## Default Methods by Mode

**Small scenarios** (--scenario):
- ground_truth (Gurobi optimal)
- direct_qpu (direct embedding)

**Traditional scales** (--scale):
- ground_truth
- direct_qpu (only for n_farms ≤ 15)
- coordinated
- decomposition methods (PlotBased, Multilevel, Louvain, Spectral)
- cqm_first methods

## Quick Troubleshooting

**"Unknown scenarios"** → Use exact names from table above

**"No D-Wave token"** → Set via `--token` or `DWAVE_API_TOKEN` env var

**Embedding timeout** → Try smaller scenario (micro_6 or micro_12)

## Output Location

Results: `qpu_benchmark_results/qpu_benchmark_<timestamp>.json`

Report: `qpu_benchmark_results/qpu_benchmark_<timestamp>.txt`

---

📖 **Full documentation**: See `QPU_SMALL_SCENARIOS_GUIDE.md`
