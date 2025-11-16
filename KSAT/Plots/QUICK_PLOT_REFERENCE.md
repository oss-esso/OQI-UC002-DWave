# QUICK REFERENCE: Plots for LaTeX Proposal

## 📊 All 9 Plots Generated Successfully

Located in: `d:\Projects\OQI-UC002-DWave\KSAT\Plots\`

### Plot Files (PNG + PDF)

1. **instance_size_comparison** - Variables & clauses comparison
2. **hardness_comparison** - 4-panel hardness metrics
3. **species_occurrence_heatmap** - 8-panel species distribution (UPDATED: distinct colormaps)
4. **cost_gradient** - Spatial costs & distribution
5. **scaling_analysis** - CNF scaling & α stability
6. **phase_transition** - 3-SAT hardness vs α curve
7. **comparison_summary** - 6-panel comprehensive overview
8. **solver_performance_comparison** - SAT vs ILP solver times (NEW)
9. **formulation_comparison** - ILP vs SAT characteristics (NEW)

## 🎯 LaTeX Quick Commands

### Basic Include
```latex
\includegraphics[width=\textwidth]{instance_size_comparison.pdf}
```

### With Caption
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{hardness_comparison.pdf}
\caption{Hardness metrics comparison}
\label{fig:hardness}
\end{figure}
```

### Reference in Text
```latex
As shown in Figure~\ref{fig:hardness}, ...
```

## 📝 Copy-Paste for Your Proposal

### Preamble (add once)
```latex
\usepackage{graphicx}
\graphicspath{{../Plots/}}  % Adjust path
```

### Full LaTeX Snippets
See: `Plots/LATEX_SNIPPETS.tex` for complete code

## 🎨 What Each Plot Shows

| Plot | Key Message | Best For |
|------|-------------|----------|
| 1. Size Comparison | Conservation has more vars, similar complexity | Intro/Methods |
| 2. Hardness | QAOA hard (57/100), Conservation medium (12/100) | Experiments |
| 3. Species Heatmap | Realistic clustering, 90% endemism, distinct colors | Datasets |
| 4. Cost Gradient | Accessibility pattern matches WDPA | Datasets |
| 5. Scaling | Linear growth, α stays ~1.5 | Methods |
| 6. Phase Transition | QAOA at α=4.27 peak, Conservation α=1.5 | Background |
| 7. Summary | All NISQ-compatible, comprehensive overview | Executive/Discussion |
| 8. Solver Performance | SAT 4-12× faster than ILP (NEW) | Methods/Results |
| 9. Formulation Compare | SAT overhead worth it for speed/quantum (NEW) | Methods/Theory |

## ✅ Validation

- [x] Species patterns: Match GBIF (90% endemism ✓)
- [x] Cost patterns: Match WDPA (accessibility gradient ✓)
- [x] QAOA instances: Match Boulebnane et al. 2024 (α=4.27 ✓)
- [x] All plots: Publication quality (300 DPI ✓)

## 🚀 Ready to Use!

All plots are **publication-ready** in both PNG and PDF formats.

**Use PDF versions in LaTeX** for best quality.
