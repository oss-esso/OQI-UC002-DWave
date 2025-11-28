# Comprehensive Benchmark Summary

**Generated:** 20251128_134739

## Configuration

- Problem sizes: [25]
- Foods per farm: 27
- Embedding timeout: 300s
- Solve timeout: 300s
- Formulations: ['CQM', 'BQM', 'SparseBQM']
- Decompositions: ['None', 'Louvain', 'PlotBased', 'Multilevel', 'Cutset', 'SpatialGrid', 'EnergyImpact']

## Results Summary

| n_farms | Formulation | Decomposition | Partitions | Embed? | Embed Time | Solve? | Solve Time | Total Time |
|---------|-------------|---------------|------------|--------|------------|--------|------------|------------|
| 25 | CQM | None | 1 | [NO] | 0.0s | [OK] | 0.0s | 0.0s |
| 25 | BQM | None | 1 | [NO] | 1076.3s | [NO] | 300.6s | 1376.9s |
| 25 | BQM | Louvain | 6 | [OK] | 1473.0s | [NO] | 300.3s | 1773.2s |
| 25 | BQM | PlotBased | 5 | [OK] | 1547.3s | [OK] | 0.3s | 1547.5s |
| 25 | BQM | Multilevel | 2 | [NO] | 967.6s | [NO] | 300.3s | 1267.9s |
| 25 | BQM | Cutset | 16 | [OK] | 224.0s | [OK] | 0.1s | 224.1s |
| 25 | BQM | SpatialGrid | 5 | [OK] | 2946.8s | [OK] | 0.6s | 2947.4s |
| 25 | BQM | EnergyImpact | 1 | [NO] | 1289.7s | [NO] | 300.1s | 1589.7s |
| 25 | SparseBQM | None | 1 | [OK] | 1.5s | [OK] | 0.0s | 1.5s |
| 25 | SparseBQM | Louvain | 5 | [OK] | 1.7s | [OK] | 0.1s | 1.8s |
| 25 | SparseBQM | PlotBased | 5 | [OK] | 1.6s | [OK] | 0.0s | 1.6s |
| 25 | SparseBQM | Multilevel | 2 | [OK] | 1.3s | [OK] | 0.0s | 1.3s |
| 25 | SparseBQM | Cutset | 14 | [OK] | 1.0s | [OK] | 0.0s | 1.0s |
| 25 | SparseBQM | SpatialGrid | 5 | [OK] | 0.9s | [OK] | 0.0s | 1.0s |
| 25 | SparseBQM | EnergyImpact | 1 | [OK] | 0.9s | [OK] | 0.0s | 0.9s |

## Statistics

- Total experiments: 15
- Successful embeddings: 11
- Successful solves: 11

## Best Configurations by Problem Size

- **25 farms**: CQM + None (total: 0.0s)