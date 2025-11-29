# Comprehensive Benchmark Summary

**Generated:** 20251129_215118

## Configuration

- Problem sizes: [25]
- Foods per farm: 27
- Embedding timeout: 300s
- Solve timeout: 60s
- Formulations: ['CQM', 'BQM']
- Decompositions: ['None', 'Louvain', 'PlotBased', 'Cutset', 'SpatialGrid']

## Results Summary

| n_farms | Formulation | Decomposition | Partitions | Embed? | Embed Time | Solve? | Solve Time | Total Time |
|---------|-------------|---------------|------------|--------|------------|--------|------------|------------|
| 25 | CQM | None | 1 | [NO] | 0.0s | [OK] | 0.0s | 0.0s |
| 25 | BQM | None | 1 | [NO] | 0.0s | [NO] | 60.1s | 60.1s |
| 25 | BQM | Louvain | 55 | [OK] | 48.2s | [OK] | 163.0s | 0.0s |
| 25 | BQM | PlotBased | 5 | [OK] | 336.1s | [NO] | 300.5s | 0.0s |
| 25 | BQM | Cutset | 84 | [OK] | 2.6s | [OK] | 0.1s | 0.0s |
| 25 | BQM | SpatialGrid | 9 | [OK] | 78.1s | [NO] | 540.5s | 0.0s |

## Statistics

- Total experiments: 6
- Successful embeddings: 4
- Successful solves: 3

## Best Configurations by Problem Size

- **25 farms**: BQM + Louvain (total: 0.0s)