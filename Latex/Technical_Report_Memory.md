# Technical Report Memory File

## Project Overview
- **Title**: Quantum-Classical Hybrid Optimization for Sustainable Food Production
- **Focus**: D-Wave quantum annealing for crop allocation optimization
- **Problem**: MILP formulation converted to CQM/BQM for quantum solvers

## Key Data Sources
1. **Benchmark Results**: `professional_plots/qpu_benchmark_*.pdf`
2. **Food Data**: 27 crops from Indonesian dataset
3. **Formulation**: Binary crop allocation with U variables for unique food tracking

## Benchmark Scales Tested
- Small: 10, 15, 50, 100 farms
- Large: 200, 500, 1000 farms
- Variables per scale: N_farms × 27 foods + 27 U variables

## Methods Compared
1. **Classical**: Gurobi (optimal baseline)
2. **Hybrid**: D-Wave Hybrid CQM Sampler
3. **QPU Decomposition Methods**:
   - PlotBased_QPU
   - Multilevel(5)_QPU
   - Multilevel(10)_QPU
   - Louvain_QPU
   - Spectral(10)_QPU
   - cqm_first_PlotBased
   - coordinated

## Key Findings from Plots
### Performance (from benchmark_comprehensive.pdf)
- Gurobi: ~0.01-0.32s across all scales
- D-Wave Hybrid CQM: ~5-12s (constant, good scaling)
- QPU methods: 30-3500s (embedding overhead)

### Solution Quality (from summary table)
- Gurobi: 0.0% gap (optimal)
- Best QPU methods: ~7-15% gap at small scale
- Coordinated: Best QPU quality at small scales
- cqm_first_PlotBased: -1.9% gap at 15 farms (beats optimal?!)

### Feasibility Issues
- coordinated: 1-23 violations at larger scales
- Most QPU methods: 0-1 violations

### Crop Selection Patterns
- Gurobi optimal: Spinach dominates (99.6% at 1000 farms)
- QPU methods: More diverse (Multilevel: 27 crops, all represented)
- This diversity may be valuable for real-world applications

## Report Structure (Chapters)
1. Executive Summary
2. Problem Formulation
3. Classical Methods
4. Quantum Computing Approach
5. Decomposition Strategies
6. Benchmark Methodology
7. Results: Performance Analysis
8. Results: Solution Quality
9. Results: Crop Allocation Patterns
10. Discussion & Conclusions
11. Future Work
12. Appendices

## Progress Tracking
- [x] Chapter 1: Executive Summary
- [x] Chapter 2: Problem Formulation
- [x] Chapter 3: Classical Methods
- [x] Chapter 4: Quantum Computing Approach
- [x] Chapter 5: Decomposition Strategies
- [x] Chapter 6: Benchmark Methodology
- [x] Chapter 7: Results - Performance
- [x] Chapter 8: Results - Quality
- [x] Chapter 9: Results - Patterns
- [x] Chapter 10: Discussion
- [x] Chapter 11: Conclusions & Future Work
- [x] Appendices (A, B, C)
- [x] Bibliography

## COMPLETED: December 2, 2025

## Final Document Location
`/Users/edoardospigarolo/Documents/OQI-UC002-DWave/Latex/Technical_Report_Comprehensive.tex`

## Document Statistics
- **12 Main Chapters** (including new Chapter 10: Comprehensive Visual Analysis)
- 3 Appendices
- **~2,826 lines of LaTeX** (~70-80 pages when compiled)
- Complete mathematical formulations
- **20+ figures with extensive captions** including all QPU benchmark plots
- Comprehensive benchmark data tables
- Crop weight sensitivity analysis with 6 figures

## Plots Included
### QPU Benchmark Plots (from professional_plots/)
1. qpu_benchmark_comprehensive.pdf - 6-panel dashboard
2. qpu_benchmark_small_scale.pdf - Small scale analysis
3. qpu_benchmark_large_scale.pdf - Large scale analysis
4. qpu_benchmark_summary_table.pdf - Complete data table
5. qpu_solution_quality_comparison.pdf - 4 quality metrics
6. qpu_solution_quality_histograms.pdf - Statistical analysis
7. qpu_solution_composition_pies.pdf - Crop distribution pies
8. qpu_solution_composition_histograms.pdf - Detailed histograms
9. qpu_solution_crop_distribution_small.pdf - Small scale crops
10. qpu_solution_crop_distribution_large.pdf - Large scale crops
11. qpu_solution_detail_100farms.pdf - 100 farms detail
12. qpu_solution_detail_500farms.pdf - 500 farms detail
13. qpu_solution_detail_1000farms.pdf - 1000 farms detail
14. qpu_solution_food_groups.pdf - Food group stacked bars
15. qpu_land_utilization_pies.pdf - Land by food group
16. qpu_solution_unique_crops_heatmap.pdf - Crop selection heatmap

### Crop Weight Analysis Plots (from crop_weight_analysis/)
17. 01_top_crop_distribution.png - Spinach dominance frequency
18. 02_benefit_heatmap.png - Attribute scores heatmap
19. 03_ranking_variability.png - Ranking box plots
20. 05_spinach_analysis.png - Why spinach wins
21. 06_parallel_coordinates.png - Multi-dimensional comparison

## Key Narrative Corrections
- ✅ Removed "Hybrid CQM success story" framing
- ✅ Emphasized that pure QPU time (26.8s) is competitive with Hybrid total time (~11s)
- ✅ Highlighted decomposition as the main contribution
- ✅ Clarified that embedding overhead (classical) is the bottleneck, not quantum
- ✅ Added extensive figure captions explaining each visualization
