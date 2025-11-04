# Structure of Modified objectives.tex

## Document Flow

```
1. Implementation of a Proof of Concept
   ├── 1.1 Notation
   │   └── 1.1.1 Problem Context
   │       ├── Sets
   │       ├── Decision variables
   │       └── Common constraints
   │
   └── 1.2 Objective Implementations
       │
       ├── 1.2.1 Linear Objective ✓ UPDATED
       │   ├── Mathematical Formulation
       │   ├── Problem Classification
       │   ├── Computational Complexity [WITH REFERENCES]
       │   └── [Advantages/Limitations REMOVED]
       │
       ├── 1.2.2 Non-Linear with Piecewise Approximation ✓ UPDATED
       │   ├── Mathematical Formulation
       │   ├── Piecewise Linear Approximation
       │   ├── Problem Classification
       │   ├── Approximation Error Analysis
       │   ├── Computational Complexity [WITH REFERENCES]
       │   ├── [Advantages/Limitations REMOVED]
       │   └── Plots
       │
       ├── 1.2.3 Fractional Non-Linear with Dinkelbach ✓ UPDATED
       │   ├── Mathematical Formulation
       │   ├── Dinkelbach's Algorithm
       │   ├── Problem Classification
       │   ├── Computational Complexity [WITH REFERENCES]
       │   └── [Advantages/Limitations REMOVED]
       │
       ├── 1.2.4 Linear-Quadratic with Synergy ✓ UPDATED
       │   ├── Mathematical Formulation
       │   ├── McCormick Linearization
       │   ├── Problem Classification
       │   ├── Computational Complexity [WITH REFERENCES]
       │   ├── [Advantages/Limitations REMOVED]
       │   └── Plots
       │
       ├── 1.2.5 BQUBO Formulation ✓ UPDATED
       │   ├── Mathematical Formulation
       │   ├── Problem Classification
       │   ├── Computational Complexity [WITH REFERENCES]
       │   ├── [Advantages/Limitations REMOVED]
       │   └── Plots
       │
       └── ⭐ NEW: 1.2.6 Comparative Analysis of Objective Formulations
           ├── Comprehensive Comparison Table
           │   ├── Problem Classification
           │   ├── Modeling Capabilities
           │   ├── Computational Performance
           │   ├── Key Advantages (all 5 formulations)
           │   ├── Key Limitations (all 5 formulations)
           │   └── Best Use Cases
           └── Selection Guidelines
   
2. Benchmarking Strategies
   └── [Unchanged]

3. Resource Estimation
   └── [Unchanged]

4. Steps to Achieve a Proof of Concept
   └── [Unchanged]

APPENDIX: References for Computational Complexity Claims
   ├── Linear MILP Complexity (5 references)
   ├── Piecewise Approximation Complexity (3 references)
   ├── Dinkelbach's Algorithm (3 references)
   ├── McCormick Linearization (2 references)
   ├── MIQP Complexity (2 references)
   └── QUBO and 0-1 ILP Complexity (4 references)
   [Total: 19 authoritative references]
```

## Key Changes Summary

### ✅ Added (Lines ~96-550)
- 19 citations to authoritative references
- Enhanced complexity descriptions with technical details
- References to specific algorithms and techniques

### ✅ Added (Lines 569-630)
- New subsection: "Comparative Analysis of Objective Formulations"
- Comprehensive comparison table with 5 formulations
- Selection guidelines for choosing formulations

### ❌ Removed (throughout)
- 5 individual "Advantages and Limitations" sections
- Redundant information now consolidated in table

### ✅ Added (Lines 1000+)
- Commented BibTeX entries for all 19 references
- Detailed notes explaining relevance of each reference

## Table Contents

The comparison table provides a complete overview across:

| Category | Information |
|----------|-------------|
| Problem Classification | Type, Variable count, Complexity class |
| Modeling Capabilities | Returns type, Non-linearity, Synergy, Approximation |
| Computational Performance | Solve time, Memory, Scalability |
| Key Advantages | Primary, Secondary, Tertiary benefits |
| Key Limitations | Primary, Secondary, Tertiary drawbacks |
| Best Use Cases | Recommended application scenarios |

## Quality Metrics

- ✓ All complexity claims backed by peer-reviewed sources
- ✓ References from top-tier journals and classic textbooks
- ✓ No informal or unreliable sources
- ✓ Comprehensive coverage of all 5 formulations
- ✓ Easy-to-read unified comparison
- ✓ Document structure preserved
- ✓ LaTeX syntax validated
- ✓ Ready for compilation

## File Size Impact

- Original: ~1050 lines
- Modified: ~1221 lines
- Net increase: ~170 lines
  - Comparison table/guidelines: +70 lines
  - Reference documentation: +250 lines
  - Removed sections: -150 lines

## Citation Distribution

| Section | Citations Added |
|---------|----------------|
| Linear MILP | 5 references |
| Piecewise NLN | 3 references |
| Dinkelbach | 3 references |
| Quadratic Synergy | 4 references |
| BQUBO | 4 references |
| **Total** | **19 references** |

---
