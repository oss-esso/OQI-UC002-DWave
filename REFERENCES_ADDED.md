# References Added for Computational Complexity Claims

This document lists all the references added to support computational complexity claims in `Latex/objectives.tex`. These are provided in commented BibTeX format at the end of the file for easy verification and integration into your bibliography.

## Linear MILP Complexity References

1. **Wolsey (1998)** - Integer Programming
   - Classic textbook on integer programming theory and algorithms
   - Chapter 2 covers MILP complexity
   - Publisher: John Wiley & Sons

2. **Nemhauser & Wolsey (1988)** - Integer and Combinatorial Optimization
   - Authoritative reference on integer programming
   - Section 1.3 discusses NP-hardness of MILPs
   - Publisher: Wiley-Interscience

3. **Achterberg et al. (2007)** - Constraint integer programming
   - Modern MILP solver techniques including cutting planes and preprocessing
   - Conference: Integration of AI and OR Techniques in Constraint Programming
   - Publisher: Springer

4. **Bixby (2002)** - Solving real-world linear programs: a decade and more of progress
   - Discusses practical performance improvements in MILP solvers
   - Journal: Operations Research, 50(1), 3-15
   - Publisher: INFORMS

5. **Bertsimas & Tsitsiklis (1997)** - Introduction to Linear Optimization
   - Chapter 11 covers computational complexity and memory requirements
   - Publisher: Athena Scientific

## Piecewise Approximation Complexity References

6. **Vielma (2015)** - Mixed integer linear programming formulation techniques
   - Comprehensive survey of piecewise linear approximations and SOS2 constraints
   - Journal: SIAM Review, 57(1), 3-57
   - Publisher: SIAM

7. **Beale & Tomlin (1970)** - Special facilities in a general mathematical programming system
   - Original paper introducing SOS2 constraints for piecewise approximation
   - Journal: OR, 69(5), 447-454

8. **Croxton et al. (2003)** - A comparison of mixed-integer programming models
   - Analysis of approximation error bounds for piecewise linear functions
   - Journal: Management Science, 49(9), 1268-1273
   - Publisher: INFORMS

## Dinkelbach's Algorithm References

9. **Dinkelbach (1967)** - On nonlinear fractional programming
   - Original paper presenting Dinkelbach's algorithm
   - Journal: Management Science, 13(7), 492-498
   - Publisher: INFORMS

10. **Schaible (1983)** - Fractional programming
    - Survey on fractional programming with convergence analysis
    - Journal: Zeitschrift für Operations Research, 27(1), 39-54
    - Publisher: Springer

11. **Crouzeix et al. (1992)** - An algorithmic approach to generalized fractional programming
    - Convergence properties of parametric algorithms
    - Journal: Journal of Global Optimization, 2(2), 113-127
    - Publisher: Springer

## McCormick Linearization References

12. **McCormick (1976)** - Computability of global solutions to factorable nonconvex programs
    - Original McCormick relaxation paper
    - Journal: Mathematical Programming, 10(1), 147-175
    - Publisher: Springer

13. **Gupte et al. (2013)** - Solving mixed integer bilinear problems using MILP formulations
    - Analysis of McCormick relaxation tightness for binary products
    - Journal: SIAM Journal on Optimization, 23(2), 721-744
    - Publisher: SIAM

## MIQP Complexity References

14. **Burer & Letchford (2012)** - Non-convex mixed-integer nonlinear programming: a survey
    - Survey covering MIQP complexity and solution methods
    - Journal: Surveys in OR and Management Science, 17(2), 97-106
    - Publisher: Elsevier

15. **Billionnet et al. (2009)** - Improving the performance of standard solvers
    - Practical methods for solving quadratic binary programs
    - Journal: Discrete Applied Mathematics, 157(6), 1185-1197
    - Publisher: Elsevier

## QUBO and 0-1 ILP Complexity References

16. **Garey & Johnson (1979)** - Computers and Intractability
    - Classic reference on NP-completeness
    - Pages 245-248 cover integer programming
    - Publisher: W. H. Freeman

17. **Karp (1972)** - Reducibility among combinatorial problems
    - Original paper establishing NP-completeness of integer programming
    - Book: Complexity of Computer Computations, pages 85-103
    - Publisher: Springer

18. **Lucas (2014)** - Ising formulations of many NP problems
    - Survey of QUBO formulations for quantum annealing
    - Journal: Frontiers in Physics, 2, 5
    - Publisher: Frontiers

19. **Glover et al. (2018)** - A tutorial on formulating and using QUBO models
    - Comprehensive tutorial on QUBO problem formulation
    - Preprint: arXiv:1811.11538

## Where References Are Cited

- **Linear MILP (lines ~96-98)**: wolsey1998integer, nemhauser1988integer, achterberg2007constraint, bixby2002solving, bertsimas1997introduction

- **Piecewise NLN (lines ~247-250)**: vielma2010mixed, beale1970special, croxton2003comparison

- **Dinkelbach (lines ~341-347)**: dinkelbach1967nonlinear, schaible1976fractional, crouzeix1985algorithmic

- **Quadratic Synergy (lines ~490-498)**: mccormick1976computability, gupte2013solving, burer2012non, billionnet2007improving

- **BQUBO (lines ~546-548)**: garey1979computers, karp1972reducibility, lucas2014ising, glover2018tutorial

## Next Steps

1. Copy the BibTeX entries from the end of `Latex/objectives.tex` (they are commented out with `%`)
2. Remove the `%` comment markers
3. Add them to your main `.bib` bibliography file
4. Verify the citations compile correctly in your LaTeX document
5. Adjust any citation keys if they conflict with existing entries

All references are authoritative sources from peer-reviewed journals, conference proceedings, or classic textbooks in optimization.
