# Comprehensive Benchmark Summary

**Generated:** 20251127_203638

## Configuration

- Problem sizes: [5]
- Foods per farm: 27
- Embedding timeout: 10s
- Solve timeout: 30s
- Formulations: ['CQM', 'BQM', 'SparseBQM']
- Decompositions: ['None', 'PlotBased']

## Results Summary

| n_farms | Formulation | Decomposition | Partitions | Embed? | Embed Time | Solve? | Solve Time | Total Time |
|---------|-------------|---------------|------------|--------|------------|--------|------------|------------|
| 5 | CQM | None | 1 | [NO] | 0.0s | [NO] | 0.0s | 0.0s |
| 5 | BQM | None | 1 | [NO] | 22.6s | [NO] | 30.0s | 52.6s |
| 5 | BQM | PlotBased | 1 | [OK] | 14.2s | [NO] | 30.0s | 44.2s |
| 5 | SparseBQM | None | 1 | [OK] | 0.1s | [OK] | 0.0s | 0.1s |
| 5 | SparseBQM | PlotBased | 1 | [OK] | 0.1s | [OK] | 0.0s | 0.1s |

## Statistics

- Total experiments: 5
- Successful embeddings: 3
- Successful solves: 2

## Best Configurations by Problem Size

- **5 farms**: SparseBQM + None (total: 0.1s)