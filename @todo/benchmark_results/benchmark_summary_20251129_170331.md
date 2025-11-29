# Comprehensive Benchmark Summary

**Generated:** 20251129_170331

## Configuration

- Problem sizes: [25]
- Foods per farm: 27
- Embedding timeout: 300s
- Solve timeout: 300s
- Formulations: ['CQM', 'BQM']
- Decompositions: ['None', 'Louvain', 'PlotBased', 'Multilevel', 'Cutset', 'SpatialGrid', 'EnergyImpact']

## Results Summary

| n_farms | Formulation | Decomposition | Partitions | Embed? | Embed Time | Solve? | Solve Time | Total Time |
|---------|-------------|---------------|------------|--------|------------|--------|------------|------------|
| 25 | CQM | None | 1 | [NO] | 0.0s | [OK] | 0.0s | 0.0s |
| 25 | BQM | None | 1 | [NO] | 0.0s | [NO] | 300.3s | 300.3s |
| 25 | BQM | Louvain | 6 | [OK] | 1288.1s | [NO] | 300.2s | 1588.3s |
| 25 | BQM | PlotBased | 5 | [OK] | 1056.6s | [NO] | 300.2s | 1356.8s |
| 25 | BQM | Multilevel | 2 | [NO] | 673.5s | [NO] | 300.2s | 973.7s |
| 25 | BQM | Cutset | 16 | [OK] | 87.1s | [OK] | 0.1s | 87.1s |
| 25 | BQM | SpatialGrid | 5 | [OK] | 1037.3s | [OK] | 0.2s | 1037.5s |
| 25 | BQM | EnergyImpact | 1 | [NO] | 514.5s | [NO] | 300.2s | 814.7s |
| 25 | SparseBQM | None | 1 | [NO] | 0.0s | [OK] | 0.0s | 0.0s |
| 25 | SparseBQM | Louvain | 5 | [OK] | 1.0s | [OK] | 0.0s | 1.1s |
| 25 | SparseBQM | PlotBased | 5 | [OK] | 0.9s | [OK] | 0.0s | 0.9s |
| 25 | SparseBQM | Multilevel | 2 | [OK] | 0.7s | [OK] | 0.0s | 0.8s |
| 25 | SparseBQM | Cutset | 14 | [OK] | 0.7s | [OK] | 0.0s | 0.7s |
| 25 | SparseBQM | SpatialGrid | 5 | [OK] | 0.7s | [OK] | 0.0s | 0.8s |
| 25 | SparseBQM | EnergyImpact | 1 | [OK] | 0.7s | [OK] | 0.0s | 0.7s |

## Statistics

- Total experiments: 15
- Successful embeddings: 10
- Successful solves: 10

## Best Configurations by Problem Size

- **25 farms**: CQM + None (total: 0.0s)