# Quick Start Guide: Reserve Design K-SAT

## Installation

### Step 1: Install Python Dependencies

```bash
# Navigate to the KSAT directory
cd KSAT

# Install basic dependencies
pip install numpy matplotlib

# Install SAT solvers (choose at least one)
pip install python-sat    # Recommended: Glucose, MiniSat, CaDiCaL
pip install z3-solver     # Alternative: Z3 SMT Solver
```

### Step 2: Verify Installation

```bash
# Run tests
python test_ksat.py
```

Expected output:
```
======================================================================
RUNNING UNIT TESTS FOR RESERVE DESIGN K-SAT
======================================================================

Test 1: Instance creation
  ✓ Random instance created successfully
  ✓ Grid instance created successfully

Test 2: Solution evaluation
  ✓ Evaluation works: cost=XX.XX, feasible=True/False
  ✓ Empty solution evaluated correctly

[... more tests ...]

======================================================================
TEST SUMMARY: 6 passed, 0 failed
======================================================================
```

## Basic Usage

### Example 1: Solve a Simple Problem

```python
from reserve_design_instance import ReserveDesignInstance
from sat_solver import ReserveDesignSATSolver

# Create a random instance
instance = ReserveDesignInstance.create_random_instance(
    num_sites=15,
    num_species=4,
    budget_fraction=0.6,
    target_coverage=2,
    seed=42
)

# Solve it
solver = ReserveDesignSATSolver(instance, 'glucose4', verbose=True)
is_sat, selected_sites, stats = solver.solve()

# Check result
if is_sat:
    print(f"✓ Solution found!")
    print(f"  Selected sites: {selected_sites}")
    print(f"  Cost: {stats['cost']:.2f}")
else:
    print("✗ No solution exists")
```

### Example 2: Find Optimal Solution

```python
# Optimize to find minimum cost
is_optimal, optimal_sites, optimal_cost, opt_stats = \
    solver.solve_with_optimization(tolerance=0.01)

if is_optimal:
    print(f"✓ Optimal solution found!")
    print(f"  Minimum cost: {optimal_cost:.2f}")
    print(f"  Budget: {instance.budget:.2f}")
    print(f"  Savings: {instance.budget - optimal_cost:.2f}")
```

### Example 3: Evaluate Solution Quality

```python
# Get detailed evaluation
evaluation = instance.evaluate_solution(selected_sites)

print(f"Solution quality:")
print(f"  Sites selected: {evaluation['num_selected']}")
print(f"  Total cost: {evaluation['total_cost']:.2f}")
print(f"  Budget used: {evaluation['budget_utilization']*100:.1f}%")
print(f"  All species covered: {evaluation['all_species_satisfied']}")

# Check each species
for sp in evaluation['species_coverage']:
    status = "✓" if sp['satisfied'] else "✗"
    print(f"  {status} {sp['species']}: {sp['achieved']}/{sp['target']}")
```

## Running the Examples

```bash
# Run all examples
python examples.py
```

This will demonstrate:
1. Basic feasibility checking
2. Cost optimization
3. Spatial grid problems
4. Comparing different SAT solvers
5. Detecting infeasible instances

## Understanding the LaTeX Document

### Compile the PDF

```bash
# Install LaTeX if needed (Windows)
# Download MiKTeX from https://miktex.org/

# Compile
pdflatex reserve_design_ksat_conversion.tex
pdflatex reserve_design_ksat_conversion.tex  # Run twice for references

# Open the PDF
start reserve_design_ksat_conversion.pdf  # Windows
# or
open reserve_design_ksat_conversion.pdf   # Mac
# or
xdg-open reserve_design_ksat_conversion.pdf  # Linux
```

The document contains:
- Complete mathematical model (Section 2)
- K-SAT conversion techniques (Section 3)
- Correctness proofs (Section 3.6)
- Python implementation guide (Section 4)
- Complexity analysis (Section 5)

## Creating Custom Instances

### Manual Instance

```python
import numpy as np

instance = ReserveDesignInstance(
    num_sites=10,
    num_species=3,
    costs=np.array([5, 3, 7, 4, 6, 2, 8, 5, 3, 4]),  # Custom costs
    presence=np.array([  # Species presence matrix (sites x species)
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1],
    ]),
    targets=np.array([3, 3, 3]),  # Need 3 sites per species
    budget=20.0,
    adjacency=[(0,1), (1,2), (2,3), (3,4), (4,5), 
               (5,6), (6,7), (7,8), (8,9)],  # Linear chain
    site_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    species_names=['Eagle', 'Bear', 'Salmon']
)
```

### Grid Instance (Spatial)

```python
# Create a spatial grid instance
instance = ReserveDesignInstance.create_grid_instance(
    grid_rows=6,
    grid_cols=6,
    num_species=5,
    seed=123
)

# Visualize the solution
from visualization import visualize_grid_solution

solver = ReserveDesignSATSolver(instance, 'glucose4')
is_sat, selected, stats = solver.solve()

if is_sat:
    visualize_grid_solution(instance, selected, 6, 6, 'my_solution.png')
```

## Choosing a SAT Solver

| Solver | Pros | Cons | Best For |
|--------|------|------|----------|
| **glucose4** | Fast, reliable | Requires PySAT | General use (recommended) |
| **minisat22** | Well-tested, stable | Slower than Glucose | Debugging, verification |
| **cadical** | Very fast, modern | Less mature | Large instances |
| **z3** | Native PB support | Can be slower | Complex constraints |

### Solver Comparison Example

```python
from sat_solver import compare_solvers

results = compare_solvers(
    instance,
    solvers=['glucose4', 'minisat22', 'z3']
)

for solver, result in results.items():
    if result['success']:
        print(f"{solver}: {result['stats']['total_time']:.3f}s")
```

## Troubleshooting

### Problem: "No module named 'pysat'"
**Solution:**
```bash
pip install python-sat
```

### Problem: "Solver returns UNSAT"
**Possible causes:**
- Budget too low
- Species targets unachievable
- Problem is genuinely infeasible

**Diagnosis:**
```python
# Check if budget is sufficient
min_cost = np.min(instance.costs) * max(instance.targets)
print(f"Minimum theoretical cost: {min_cost:.2f}")
print(f"Available budget: {instance.budget:.2f}")

# Check species achievability
for j in range(instance.num_species):
    available = np.sum(instance.presence[:, j] > 0)
    needed = instance.targets[j]
    print(f"Species {j}: need {needed}, have {available}")
```

### Problem: Encoding/solving is slow
**Solutions:**
- Reduce problem size
- Use glucose4 or cadical (faster)
- Increase tolerance in optimization
- Use binary encoding instead of sequential counter

### Problem: Out of memory
**Solutions:**
- Reduce budget constraint
- Use sparse encoding
- Process in batches

## Advanced Features

### Export to DIMACS Format

```python
from sat_encoder import ReserveDesignSATEncoder

encoder = ReserveDesignSATEncoder(instance)
cnf = encoder.encode()

# Export to standard DIMACS format
cnf.to_file('problem.cnf')

# Now can use any external SAT solver
# glucose problem.cnf solution.txt
# minisat problem.cnf solution.txt
```

### Custom Optimization Bounds

```python
# Try different cost bounds
for budget in [10, 15, 20, 25, 30]:
    is_sat, selected, stats = solver.solve(objective_bound=budget)
    if is_sat:
        print(f"Budget {budget}: SAT (cost={stats['cost']:.2f})")
    else:
        print(f"Budget {budget}: UNSAT")
```

### Analyze Encoding Size

```python
from sat_encoder import ReserveDesignSATEncoder

encoder = ReserveDesignSATEncoder(instance, verbose=True)
cnf = encoder.encode()

stats = encoder.get_encoding_statistics()
print(f"Variables: {stats['num_variables']}")
print(f"Clauses: {stats['num_clauses']}")
print(f"Site variables: {stats['num_site_vars']}")
print(f"Auxiliary variables: {stats['num_auxiliary_vars']}")
print(f"Average clause length: {stats['average_clause_length']:.2f}")
```

## Performance Tips

1. **Start small**: Test with 10-20 sites first
2. **Use Glucose4**: Generally fastest for most instances
3. **Set tolerance**: For optimization, 0.01-0.1 is usually sufficient
4. **Monitor encoding**: Large encodings may need different strategies
5. **Batch processing**: For many instances, reuse encoder structure

## Next Steps

1. ✅ Read the LaTeX document for theoretical understanding
2. ✅ Run `test_ksat.py` to verify installation
3. ✅ Run `examples.py` to see demonstrations
4. ✅ Try solving your own problems
5. ✅ Experiment with different solvers and parameters
6. ✅ Visualize solutions with `visualization.py`

## Getting Help

- Check `README.md` for detailed API reference
- Read `SUMMARY.md` for implementation overview
- Review examples in `examples.py`
- Consult LaTeX document for mathematical details

## Citation

If you use this in research:

```bibtex
@misc{reserve_design_ksat_2025,
  title={Reserve Design Problem: K-SAT Encoding and Classical Solving},
  author={Your Name},
  year={2025},
  note={Based on Justiniano-Albarracin et al. (2018)}
}
```

---

**Questions?** Open an issue or consult the documentation!
