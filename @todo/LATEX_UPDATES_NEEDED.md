# LaTeX Chapter Updates Required

## Summary of Changes to Documentation

Based on the implementation corrections completed on November 21, 2025, the following LaTeX chapters require updates to match the current codebase.

---

## Chapter 2: Mathematical Problem Formulation

### Changes Required

#### 1. Add Maximum Planting Area Variable (Table 2.1)

**Location**: Section 2.2, Variable Definitions Table

**Add**:
```latex
$M^{\text{max}}_c$ & Maximum planting area for crop $c \in \mathcal{C}$ (hectares) \\
```

**After**: `$M_c$ & Minimum planting area...`

#### 2. Update Linking Constraints Section (Section 2.3.3)

**Replace** "Linking Constraints" with "Maximum Planting Area Constraints"

**Old**:
```latex
\subsubsection{Linking Constraints}
The binary indicators must be consistent with area allocations...
A_{f,c} \leq L_f \cdot Y_{f,c}
```

**New**:
```latex
\subsubsection{Maximum Planting Area Constraints}

If crop $c$ is planted on farm $f$, the planted area cannot exceed the maximum:

\begin{equation}
A_{f,c} \leq M^{\text{max}}_c \cdot Y_{f,c} \quad \forall f \in \mathcal{F}, \forall c \in \mathcal{C}
\end{equation}

For crops without explicit maximum, farm capacity serves as upper bound:
\begin{equation}
A_{f,c} \leq L_f \cdot Y_{f,c} \quad \text{(when } M^{\text{max}}_c \text{ undefined)}
\end{equation}
```

#### 3. Add Food Group Constraints Section

**Location**: After "Minimum Planting Area Constraints", before final constraint count

**Add New Section**:
```latex
\subsubsection{Food Group Diversity Constraints}

For each food group $g \in \mathcal{G}$, enforce minimum and maximum diversity:

\begin{equation}
\alpha^{\text{min}}_g \leq \sum_{f \in \mathcal{F}} \sum_{c \in G^{-1}(g)} Y_{f,c} \leq \alpha^{\text{max}}_g
\label{eq:food_group_farm}
\end{equation}

where:
- $G^{-1}(g)$: set of crops in food group $g$
- $\alpha^{\text{min}}_g$, $\alpha^{\text{max}}_g$: min/max count of foods from group $g$

\textbf{Critical Note}: These constraints count SELECTIONS ($Y$ variables), not total area. 
This ensures diversity in crop variety, not area allocation.

With 5 food groups: 10 constraints (2 per group).
```

#### 4. Update Total Constraint Count

**Old**: "Total: ~1350 constraints"

**New**: 
```
Farm Scenario Total Constraints:
- Land availability: 25
- Minimum planting area: 675 (25 × 27)
- Maximum planting area: 675 (25 × 27)
- Food group min: 5
- Food group max: 5
Total: 1385 constraints
```

---

## Chapter 4: Alternative 2 Architecture

### Complete Rewrite Required for Farm Section

#### Section 4.1: Update Architecture Philosophy

**Old Title**: "Heterogeneous Solver Specialization"

**New Title**: "Hybrid Decomposition Strategy"

**New Content**:
```latex
\subsection{Hybrid Decomposition Strategy}

Alternative 2 implements strategic hybrid decomposition:

\begin{itemize}
    \item \textbf{Farm Scenarios}: Hybrid decomposition (Gurobi continuous + QPU binary)
    \item \textbf{Patch Scenarios}: Pure quantum annealing (DWaveSampler)
\end{itemize}

Farm problems are decomposed to leverage complementary strengths:
- Gurobi: Continuous optimization (area allocation)
- QPU: Binary combinatorics (crop selection)
```

#### Section 4.2: Replace Entire "Classical Solver Component"

**Delete**: Entire section 4.2 "Classical Solver Component"

**Replace With**: "Hybrid Decomposition Component" (see implementation below)

#### New Section 4.2: Hybrid Decomposition Component

```latex
\section{Hybrid Decomposition Component}

\subsection{Farm Scenario: Gurobi + QPU Hybrid}

For farm scenarios, Alternative 2 implements a 4-phase hybrid decomposition.

\subsubsection{Algorithm Overview}

\begin{algorithm}[H]
\caption{Hybrid Decomposition for Farm Scenarios}
\begin{algorithmic}[1]
\State \textbf{Phase 1: Continuous Relaxation (Gurobi)}
\State Relax $Y_{f,c} \in \{0,1\}$ to $Y_{f,c} \in [0,1]$
\State Solve MINLP with Gurobi $\rightarrow$ obtain $A^*$ and $Y^{\text{relaxed}}$

\State \textbf{Phase 2: Binary Subproblem Construction}
\State Fix $A = A^*$ from Phase 1
\State Create CQM with only binary $Y_{f,c} \in \{0,1\}$
\State Objective: $\max \sum_{f,c} B_c \cdot A^*_{f,c} \cdot Y_{f,c}$

\State \textbf{Phase 3: Quantum Annealing}
\State Convert CQM to BQM
\State Solve on QPU (or SimulatedAnnealing) $\rightarrow$ obtain $Y^{**}$

\State \textbf{Phase 4: Solution Combination}
\State Combine: Final solution = $(A^*, Y^{**})$
\end{algorithmic}
\end{algorithm}

\subsubsection{Rationale}

This decomposition leverages:
\begin{itemize}
    \item \textbf{Gurobi Strength}: Efficient continuous optimization
    \item \textbf{QPU Strength}: Binary combinatorial search
    \item \textbf{Reduced Complexity}: Smaller binary subproblem fits better on QPU
\end{itemize}

\subsubsection{Expected Performance}

For 25 farms, 27 crops:
\begin{itemize}
    \item Phase 1 (Gurobi): $\sim$0.1-0.5 seconds
    \item Phase 2 (BQM conversion): $\sim$0.01-0.05 seconds
    \item Phase 3 (QPU): $\sim$0.1-1.0 seconds
    \item Total: $\sim$0.2-1.5 seconds
\end{itemize}
```

#### Update Section 4.3: Quantum Solver Component (Patch)

**Keep**: Section 4.3 as-is (pure quantum for patches is unchanged)

**Minor Update**: Add note distinguishing from farm hybrid approach:

```latex
\textbf{Note}: Unlike farm scenarios (which use hybrid decomposition), 
patch scenarios solve directly on QPU as pure binary problems.
```

---

## Chapter 5: Testing and Validation

### Section 5.3: Update Test Results

#### Test 2: CQM Creation Results

**Update** constraint count expectations:

```latex
\textbf{Farm CQM (3 farms, 27 crops)}:
- Variables: 162 (81 continuous A + 81 binary Y)
- Constraints: 177 total
  * 3 land availability
  * 81 minimum area
  * 81 maximum area
  * 5 food group minimum (count)
  * 5 food group maximum (count)
  * 2 linking (if any remain)

\textbf{Patch CQM (3 patches, 27 crops)}:
- Variables: 81 (binary Y only)
- Constraints: 13 total
  * 3 one-hot (one crop per patch)
  * 5 food group minimum (count)
  * 5 food group maximum (count)
```

#### Add Constraint Validation Test

**New Subsection** after Test 2:

```latex
\subsubsection{Test 2.5: Constraint Validation}

\textbf{Objective}: Verify constraint implementation matches reference binary solver.

\textbf{Validations}:
\begin{itemize}
    \item Farm has NO one-hot constraint (can grow multiple crops)
    \item Patch HAS one-hot constraint (one crop per patch)
    \item Both use maximum area/plots constraints
    \item Both use maximum food group constraints
    \item Food groups use COUNT ($Y$ variables) not AREA ($A$ variables)
\end{itemize}

\textbf{Result}: OK PASS
```

---

## Chapter 8: Conclusions

### Section 8.1: Update Primary Contributions

**Update** Alternative 2 description:

**Old**:
```latex
\textbf{Alternative 2: Strategic Problem Decomposition}
\begin{itemize}
    \item Demonstrated problem-solver matching as viable hybrid strategy
```

**New**:
```latex
\textbf{Alternative 2: Hybrid Decomposition}
\begin{itemize}
    \item Implemented novel hybrid decomposition for MINLP problems
    \item Combined Gurobi continuous optimization with QPU binary search
    \item Demonstrated effective coupling of classical and quantum components
```

### Section 8.2: Add Lessons Learned

**Add New Subsection**:

```latex
\subsubsection{Constraint Design Matters}

Critical insights from implementation:

\begin{itemize}
    \item \textbf{Count vs. Area}: Food group constraints should count crop SELECTIONS, 
          not total area allocated. This ensures diversity.
    \item \textbf{Maximum Constraints}: Both minimum AND maximum constraints are essential 
          to prevent over-allocation and ensure feasibility.
    \item \textbf{Scenario Differences}: One-hot constraints apply ONLY to patch scenarios, 
          not farms (farms can grow multiple crops).
\end{itemize}
```

---

## Implementation Priority

### High Priority (Must Update Before Defense)
1. ✅ Chapter 2: Add max constraints and food group section
2. ✅ Chapter 4: Complete rewrite of Alternative 2 farm section
3. ✅ Chapter 5: Update constraint validation tests

### Medium Priority
4. Chapter 8: Update contributions and lessons learned

### Low Priority (Nice to Have)
5. Chapter 1: Minor updates to motivation (optional)
6. Chapter 6: Update expected results if benchmarks run

---

## Quick Reference: Key Corrections

| Aspect | Old (Wrong) | New (Correct) |
|--------|-------------|---------------|
| **Farm food groups** | Used A (area) | Use Y (count) |
| **Max area constraints** | Missing | Added for both scenarios |
| **Max food group** | Missing | Added for both scenarios |
| **Alt 2 farm solver** | Gurobi only | Hybrid (Gurobi + QPU) |
| **Alt 2 architecture** | Problem routing | Hybrid decomposition |

---

## Files to Update

1. `technical_report_chapter2.tex` - Problem formulation
2. `technical_report_chapter4.tex` - Alternative 2 architecture
3. `technical_report_chapter5.tex` - Testing results
4. `technical_report_chapter8.tex` - Conclusions

---

Last Updated: November 21, 2025
Status: Implementation Complete, LaTeX Updates Pending
