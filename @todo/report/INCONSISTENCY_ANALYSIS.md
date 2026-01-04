# Inconsistency Analysis of content_report.tex

This report details all identified inconsistencies in the Phase 3 scientific report, categorized by type and severity.

## Summary of Critical Inconsistencies

1. **Abstract vs. Study 3 Contradiction**: The abstract claims QPU achieves "solution quality within 10 to 15% of classical optimal" (implying QPU is worse), while Study 3 claims QPU achieves "3.80× higher benefit than Gurobi" (implying QPU is better). These are mutually exclusive claims.

2. **Figure Paths Do Not Exist**: All figures reference `images/Plots/...` but this directory does not exist in the report folder. Actual figures reside in `professional_plots/` and `Phase3Report/Plots/`.

3. **Hardware Specification Contradictions**: Section 2.3 specifies "CERN's SWAN with 4 CPU cores, 16GB RAM, Tesla T15 GPU" while Section 3.4.1 specifies "Intel Core i7-12700H (14 cores, 20 threads), 32 GB RAM".

---

## 1. Figure vs. Context Inconsistencies

### Inconsistency 1.1: All Figure Paths Invalid
- **Severity:** Critical
- **Location:** All figure environments throughout the document
- **Description:** Every `\includegraphics` command references paths under `images/Plots/` (e.g., `images/Plots/01_top_crop_distribution.png`), but this directory does not exist in the `@todo/report/` folder. The actual figures exist in:
  - `c:\Users\Edoardo\Documents\EPFL\OQI-UC002-DWave\professional_plots\`
  - `c:\Users\Edoardo\Documents\EPFL\OQI-UC002-DWave\Phase3Report\Plots\`
  
  **Affected figures:**
  - `images/Plots/01_top_crop_distribution.png` (Fig 4, line 1422)
  - `images/Plots/02_benefit_heatmap.png` (Fig 5, line 1431)
  - `images/Plots/violation_gap_analysis.pdf` (Fig 6, line 1440)
  - `images/Plots/violation_impact_assessment.pdf` (Fig 7, line 1465)
  - `images/Plots/quantum_advantage_objective_scaling.pdf` (Fig 8, line 1527)
  - `images/Plots/quantum_advantage_split_analysis.pdf` (Fig 9, line 1570)
  - `images/Plots/comprehensive_scaling.png` (Fig 10, line 1752)
  - `images/Plots/variable_count_scaling_analysis.png` (Fig 11, line 1759)
  - `images/Plots/plot_time_vs_vars.png` (Fig 12, line 1820)
  - `images/Plots/qpu_solution_composition_pies.png` (Fig 13, line 1859)
  - `images/Plots/qpu_solution_composition_histograms.png` (Fig 14, line 1871)
  - `images/Plots/plot_gap_speedup_vs_vars.png` (Fig 15, line 1885)
  - `images/Plots/plot_solution_quality_vs_vars.png` (Fig 16, line 1897)

### Inconsistency 1.2: Caption Mismatch in Figure 6 (violation_gap_analysis)
- **Severity:** Major
- **Location:** Section 3.2, Figure 6 (`\label{fig:violation_gap_analysis}`)
- **Description:** The caption references "pure QPU time linear scaling at 0.1489ms/var" but this specific rate is not consistently cited elsewhere in the document. The caption also mentions "6-Family average 278.8% gap vs 27-Food 343.6% gap" which should be cross-verified with Table 3.6 data.

### Inconsistency 1.3: Figure 7 Caption vs. Table Data
- **Severity:** Major
- **Location:** Section 3.2, Figure 7 (`\label{fig:violation_impact_assessment}`)
- **Description:** The caption states "violations account for only 7% of the objective gap, with remaining 93% due to decomposition approximation errors" but this specific percentage breakdown is not supported by any table in the document.

### Inconsistency 1.4: Referenced Figure Does Not Exist in Text
- **Severity:** Minor
- **Location:** Section 4.3.4, line ~1816
- **Description:** The text references `\cref{fig:time_comparison}` but this label doesn't appear to match any figure in the document. The surrounding context discusses Figure showing "wall-clock time comparison on logarithmic scale" but the actual figure label appears to be different.

---

## 2. Report vs. Workspace Inconsistencies

### Inconsistency 2.1: HybridGrid Method Mentioned But Not In Results
- **Severity:** Major
- **Location:** Section 1.1.2 (Study 2 description), Section 2.5.8
- **Description:** The introduction lists "eight decomposition strategies (Direct QPU, PlotBased, Multilevel, Louvain, Spectral, Coordinated, CQM-First PlotBased, HybridGrid)" and Section 2.5.8 describes HybridGrid as "the best-performing pure QPU approach in our benchmarks." However:
  - Table 2.5 (`tab:method_comparison`) only lists 8 methods but excludes HybridGrid from the comparison
  - Table 3.2 (`tab:decomposition_methods`) lists only 7 methods, excluding HybridGrid
  - Table 3.5 (`tab:quality_1000farms`) does not include HybridGrid results
  - The workspace contains HybridGrid implementations in `plot_config.py` (HybridGrid(5,9)_QPU, HybridGrid(10,9)_QPU) and `qpu_benchmark.py`, but no results appear in the report tables.

### Inconsistency 2.2: 16,308% MIP Gap Claim Unverified
- **Severity:** Major
- **Location:** Section 3.4.3, Table 3.7 (`tab:gurobi_struggles`)
- **Description:** The report claims "average MIP gaps of 16,308%" with "Max MIP Gap" of "176,411%" and "352,822%" for large instances. These extraordinarily high values are not found in any of the workspace data files (`benchmark_*.json`, `gurobi_*.json`). The claim should be verified against actual solver logs or marked as requiring verification.

### Inconsistency 2.3: Gurobi Version Discrepancy
- **Severity:** Minor
- **Location:** Section 2.3 vs Section 3.4.1
- **Description:** Section 2.3 states "Gurobi 11.0.3" while Section 3.4.1 states "Gurobi 12.0.1". Different Gurobi versions may produce different performance characteristics.

### Inconsistency 2.4: Script Names Not Verified
- **Severity:** Minor
- **Location:** Throughout document
- **Description:** The workspace contains `analyze_violation_gap.py` which appears to match analysis described in the report, but the report doesn't explicitly cite this or other analysis scripts. For reproducibility, specific script names should be referenced.

---

## 3. Internal Report Inconsistencies

### Inconsistency 3.1: CRITICAL - Abstract vs. Study 3 Solution Quality Contradiction
- **Severity:** Critical
- **Location:** Abstract (lines 74-78) vs. Section 3.4 (lines 1500-1710)
- **Description:** 
  
  **Abstract claims (line 77):**
  > "We achieve solution quality within 10 to 15% of classical optimal while maintaining constraint feasibility."
  
  This implies QPU solutions are **10-15% worse** than Gurobi optimal.
  
  **Study 3 claims (Section 3.4.2, lines 1523-1560):**
  > "The QPU consistently achieves **3.80× higher benefit values** than Gurobi across all 13 benchmark scenarios."
  
  This implies QPU solutions are **280% better** than Gurobi.
  
  **These claims are mutually exclusive.** The document later explains that Gurobi times out and cannot find optimal solutions for Formulation B (rotation problems), so the 3.80× advantage is comparing QPU to Gurobi's timeout-limited suboptimal solution, not to true optimal. However, the abstract's "10-15% of classical optimal" claim appears to refer to Formulation A (binary allocation), while the 3.80× claim refers to Formulation B. This distinction is not clear in the abstract.
  
  **Recommended fix:** The abstract should clearly distinguish between the two formulations and their different results:
  - Formulation A: QPU achieves 10-15% gap from classical optimal
  - Formulation B: QPU achieves 3.80× higher benefit than timeout-limited classical solver

### Inconsistency 3.2: Hardware Configuration Contradiction
- **Severity:** Critical
- **Location:** Section 2.3 vs. Section 3.4.1
- **Description:**
  
  **Section 2.3 (lines 207-208):**
  > "All experiments were conducted on CERN's SWAN with 4 CPU cores, 16GB of RAM and a Tesla T15 GPU"
  
  **Section 3.4.1 (lines 1513-1517):**
  > "Solver: Gurobi 12.0.1 (academic license)"
  > "CPU: Intel Core i7-12700H (14 cores, 20 threads)"
  > "Memory: 32 GB RAM"
  
  These describe entirely different hardware configurations. Either the experiments were run on different machines for different sections (which should be explicitly stated), or one specification is incorrect.

### Inconsistency 3.3: Timeout Value Inconsistency
- **Severity:** Major
- **Location:** Section 2.3 vs. Section 3.4.1 and throughout
- **Description:**
  
  **Section 2.3 (line 199):**
  > "We configured Gurobi with a **100 to 300 second timeout** per problem instance"
  
  **Section 3.4.1 (line 1516):**
  > "Timeout: 300 seconds per scenario"
  
  **Table 3.7 (line ~1592):**
  > Uses 60s timeout reference in figure caption (line 1529)
  
  The timeout value should be consistent throughout. If different timeouts were used for different experiments, this should be explicitly documented.

### Inconsistency 3.4: Speedup Claims Inconsistency
- **Severity:** Major
- **Location:** Abstract vs. Section 4.3.4
- **Description:**
  
  **Abstract (lines 75-76):**
  > "our hierarchical quantum-classical decomposition achieves 5 to 9× speedups for problems with 25 to 100 farms"
  
  **Section 4.3.4 (lines ~1816-1820):**
  > "For the smallest instances (5 farms), the quantum methods complete in approximately 20 to 24 seconds, representing a **12 to 15× speedup**"
  
  These speedup figures (5-9× vs 12-15×) don't align and refer to different problem sizes. The abstract should be consistent with the detailed results.

### Inconsistency 3.5: QPU Time Claims Vary
- **Severity:** Major
- **Location:** Abstract vs. Table 3.3 vs. Table 3.8
- **Description:**
  
  **Abstract (line 76):**
  > "Pure QPU access time scales linearly with problem size, remaining under 30 seconds even for 100-farm instances with 1,800 decision variables"
  
  **Table 3.3 (`tab:qpu_time_scaling`, line ~1318):**
  - 100 farms: Pure QPU = 2.15s (not 30s)
  - 1,000 farms: Pure QPU = 21.78s
  
  **Table 3.8 (`tab:timing_comparison`, line ~1632):**
  > Combined: QPU Pure = 10.88s total across 13 scenarios
  
  The "under 30 seconds" claim is accurate but the specific values cited elsewhere show much lower times. Clarification needed.

### Inconsistency 3.6: Constraint Violation Rate Discrepancy
- **Severity:** Minor
- **Location:** Section 3.4.3 vs. Section 3.4.5
- **Description:**
  
  **Section 3.4.3 (line 1602):**
  > "average 21.9% violation rate"
  
  **Section 3.4.5 (Table `tab:violation_impact`, line ~1669):**
  > "Overall violation rate: 24.2%"
  
  The violation rate is cited as both 21.9% and 24.2% in different parts of the same section.

### Inconsistency 3.7: Duplicate SDG Subsection Header
- **Severity:** Minor
- **Location:** Lines 90-91
- **Description:** The `\subsubsection*{Relevant SDGs}` header appears twice consecutively:
  ```latex
  \subsubsection*{Relevant SDGs}
      % Instructions to be deleted before submission
      \textit{Please list the relevant UN SDGs}
      \subsubsection*{Relevant SDGs}
  ```

### Inconsistency 3.8: "Needs checking" Annotation Left in Final Text
- **Severity:** Minor
- **Location:** Section 4 header (line ~1776)
- **Description:** The text contains:
  > "**needs double (maybe triple) checking**"
  
  This annotation should be removed before final submission.

### Inconsistency 3.9: Notes Section in Abstract Area
- **Severity:** Minor
- **Location:** Lines 79-84
- **Description:** A "Notes" section with TODO items remains in the document:
  > "Notes: All references will be added later, each mention of [add ref] means I have it"
  > "figures are still not the correct ones"
  > "the section about advantage could use more testing"
  
  These should be removed before submission.

---

## 4. Additional Observations

### Observation 4.1: Commented-Out Figures
Several figures are commented out in the LaTeX source (lines 1654-1663), suggesting they were removed or replaced but the references may not have been updated throughout the document.

### Observation 4.2: Missing Reference Links
Multiple `[add ref]` placeholders remain in the document (lines 161, 183, 199, 490) indicating incomplete bibliography.

### Observation 4.3: Incomplete Appendix
The "Appendix: The Spinach Issue" (line ~2290) contains only the header "**needs checking**" with no actual content, though extensive commented-out figure analysis exists later in the file.

---

## Recommendations

1. **Immediate Critical Fixes:**
   - Resolve the abstract vs. Study 3 solution quality contradiction by clearly distinguishing Formulation A and B results
   - Create the `images/Plots/` directory and copy required figures, or update all paths to correct locations
   - Clarify which hardware was used for which experiments

2. **Major Fixes:**
   - Add HybridGrid results to comparison tables or remove claims about it being "best-performing"
   - Verify and cite sources for the 16,308% MIP gap claim
   - Standardize timeout values and document any variations
   - Reconcile speedup claims between abstract and body

3. **Minor Fixes:**
   - Remove duplicate SDG header
   - Remove "needs checking" annotations
   - Remove Notes section from abstract area
   - Complete bibliography references
   - Standardize Gurobi version reference
   - Reconcile violation rate percentages (21.9% vs 24.2%)

---

*Analysis generated: January 4, 2026*
*Document analyzed: content_report.tex (2793 lines)*
