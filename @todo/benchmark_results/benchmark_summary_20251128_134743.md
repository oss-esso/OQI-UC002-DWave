# Comprehensive Benchmark Summary

**Generated:** 20251128_134743

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
| 25 | BQM | None | 1 | [NO] | 347.5s | [NO] | 300.2s | 647.7s |
| 25 | BQM | Louvain | 6 | [OK] | 1563.1s | [NO] | 300.5s | 1863.6s |
| 25 | BQM | PlotBased | 5 | [OK] | 1430.4s | [OK] | 0.3s | 1430.6s |
| 25 | BQM | Multilevel | 2 | [NO] | 845.4s | [NO] | 300.7s | 1146.1s |
| 25 | BQM | Cutset | 16 | [OK] | 479.1s | [OK] | 0.1s | 479.2s |
| 25 | BQM | SpatialGrid | 5 | [OK] | 3313.8s | [OK] | 0.2s | 3314.0s |
| 25 | BQM | EnergyImpact | 1 | [NO] | 1280.2s | [NO] | 300.1s | 1580.4s |
| 25 | SparseBQM | None | 1 | [OK] | 1.0s | [OK] | 0.0s | 1.0s |
| 25 | SparseBQM | Louvain | 5 | [OK] | 1.0s | [OK] | 0.0s | 1.0s |
| 25 | SparseBQM | PlotBased | 5 | [OK] | 1.0s | [OK] | 0.0s | 1.0s |
| 25 | SparseBQM | Multilevel | 2 | [OK] | 1.3s | [OK] | 0.0s | 1.3s |
| 25 | SparseBQM | Cutset | 14 | [OK] | 1.0s | [OK] | 0.0s | 1.1s |
| 25 | SparseBQM | SpatialGrid | 5 | [OK] | 0.8s | [OK] | 0.0s | 0.9s |
| 25 | SparseBQM | EnergyImpact | 1 | [OK] | 0.8s | [OK] | 0.0s | 0.8s |

## Statistics

- Total experiments: 15
- Successful embeddings: 11
- Successful solves: 11

## Best Configurations by Problem Size

- **25 farms**: CQM + None (total: 0.0s)