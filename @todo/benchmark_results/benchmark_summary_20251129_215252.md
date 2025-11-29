# Comprehensive Benchmark Summary

**Generated:** 20251129_215252

## Configuration

- Problem sizes: [25]
- Foods per farm: 27
- Embedding timeout: 30s
- Solve timeout: 30s
- Formulations: ['CQM', 'BQM']
- Decompositions: ['None', 'Louvain', 'PlotBased']

## Results Summary

| n_farms | Formulation | Decomposition | Partitions | Embed? | Embed Time | Solve? | Solve Time | Total Time |
|---------|-------------|---------------|------------|--------|------------|--------|------------|------------|
| 25 | CQM | None | 1 | [NO] | 0.0s | [OK] | 0.0s | 0.0s |
| 25 | BQM | None | 1 | [NO] | 0.0s | [NO] | 30.1s | 30.1s |
| 25 | BQM | Louvain | 55 | [OK] | 49.0s | [OK] | 192.5s | 241.6s |
| 25 | BQM | PlotBased | 5 | [NO] | 157.4s | [NO] | 150.4s | 307.8s |

## Statistics

- Total experiments: 4
- Successful embeddings: 1
- Successful solves: 2

## Best Configurations by Problem Size

- **25 farms**: CQM + None (total: 0.0s)