# Comprehensive Benchmark Summary

**Generated:** 20251128_134725

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
| 25 | BQM | None | 1 | [NO] | 351.5s | [NO] | 300.1s | 651.6s |
| 25 | BQM | Louvain | 6 | [OK] | 1541.1s | [NO] | 300.3s | 1841.4s |
| 25 | BQM | PlotBased | 5 | [OK] | 1522.8s | [OK] | 0.3s | 1523.1s |
| 25 | BQM | Multilevel | 2 | [NO] | 917.4s | [NO] | 300.3s | 1217.8s |
| 25 | BQM | Cutset | 16 | [OK] | 480.6s | [OK] | 0.1s | 480.7s |
| 25 | BQM | SpatialGrid | 5 | [OK] | 1395.5s | [OK] | 0.4s | 1395.9s |
| 25 | BQM | EnergyImpact | 1 | [NO] | 3088.4s | [NO] | 300.2s | 3388.5s |
| 25 | SparseBQM | None | 1 | [OK] | 2.5s | [OK] | 0.0s | 2.5s |
| 25 | SparseBQM | Louvain | 5 | [OK] | 3.0s | [OK] | 0.1s | 3.1s |
| 25 | SparseBQM | PlotBased | 5 | [OK] | 2.4s | [OK] | 0.0s | 2.5s |
| 25 | SparseBQM | Multilevel | 2 | [OK] | 2.5s | [OK] | 0.1s | 2.6s |
| 25 | SparseBQM | Cutset | 14 | [OK] | 2.9s | [OK] | 0.0s | 2.9s |
| 25 | SparseBQM | SpatialGrid | 5 | [OK] | 2.5s | [OK] | 0.1s | 2.5s |
| 25 | SparseBQM | EnergyImpact | 1 | [OK] | 2.6s | [OK] | 0.0s | 2.6s |

## Statistics

- Total experiments: 15
- Successful embeddings: 11
- Successful solves: 11

## Best Configurations by Problem Size

- **25 farms**: CQM + None (total: 0.0s)