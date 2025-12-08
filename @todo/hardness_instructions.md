# ILP Hardness Diagnostic Prompt

This prompt instructs a coding agent to run four diagnostic checks (Sections 1–4) on an Integer Linear Program (ILP) to estimate how "hard" it is for a branch-and-cut solver (e.g., Gurobi). For each check, the agent should run the commands or code, collect the outputs, and save a short JSON report with the measured metrics.

---

## Inputs (agent must be given)

* Path to ILP file (LP, MPS, or a Gurobi model object) — `input_path`
* Optional: known incumbent integer solution (if available) — `incumbent_solution` (JSON or Gurobi solution file)
* Working folder for outputs — `out_dir`
* Python environment with Gurobi installed and accessible as `gurobipy`

---

## Output (what agent must produce)

* `out_dir/report.json` — a JSON file summarizing metrics from the 4 checks.
* `out_dir/log.txt` — console log of commands and solver outputs.
* Optional auxiliary files: `out_dir/root_solution.csv`, `out_dir/coeff_stats.json`.

---

# SECTION 1 — Integrality (root) gap

**Goal:** compute LP relaxation objective, integer objective, and integrality gap.

**Steps for the agent (Python + Gurobi):**

1. Load the model from `input_path`.
2. Optimize the MIP to obtain an incumbent (if none, set `int_obj = null`).
3. Create and solve the LP relaxation (use `model.relax()` or copy & set integer vars to `GRB.CONTINUOUS`).
4. Record `lp_obj` and `int_obj`. Compute integrality gap for minimization as:

   ```text
   gap = (int_obj - lp_obj) / max(1.0, abs(int_obj))
   ```

   (If `int_obj` is null, report `gap = null` and still provide `lp_obj`.)

**Deliverables:**

* `report.json` fields: `lp_obj`, `int_obj`, `integrality_gap`.
* A one-line human-readable summary in `log.txt`.

**Example code (agent may adapt):**

```python
import gurobipy as gp
from gurobipy import GRB
import json

m = gp.read(input_path)  # or gp.Model(input_path)
# 1) solve MIP to get integer incumbent
m.optimize()
int_obj = m.ObjVal if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT and m.SolCount>0 else None

# 2) LP relaxation
rel = m.relax()
rel.optimize()
lp_obj = rel.ObjVal if rel.status == GRB.OPTIMAL else None

if int_obj is not None and lp_obj is not None:
    gap = (int_obj - lp_obj) / max(1.0, abs(int_obj))
else:
    gap = None

report = {'lp_obj': lp_obj, 'int_obj': int_obj, 'integrality_gap': gap}
with open(out_dir+'/report.json','w') as f:
    json.dump(report, f, indent=2)
```

---

# SECTION 2 — Fractionality & root solution structure

**Goal:** count how many variables are fractional at the LP root and identify which variables/constraints are most involved.

**Steps for the agent:**

1. Use the LP relaxation solved in Section 1 (`rel` model).
2. Extract variable values and count fractionals: `abs(x - round(x)) > 1e-8` (tolerance can be adjusted).
3. Save `root_solution.csv` with columns: `var_name, var_type, value, fractional_flag`.
4. Summarize: `fractional_count`, `fractional_ratio` (fractional_count / total_vars).
5. Optionally compute a small variable-to-constraint participation map for the top-K fractional variables: list the constraints (IDs) in which each appears and the absolute coefficient magnitude.

**Deliverables:**

* `root_solution.csv`
* `report.json` additions: `fractional_count`, `fractional_ratio`, `top_fractional_vars` (list of var names + quick stats)

**Example code snippet:**

```python
vars_rel = rel.getVars()
frac_tol = 1e-8
rows = []
fractional_count = 0
for v in vars_rel:
    val = v.x
    is_frac = abs(val - round(val)) > frac_tol
    rows.append((v.VarName, v.VType, val, is_frac))
    fractional_count += int(is_frac)
# write CSV (use csv module or pandas)

report.update({
  'fractional_count': fractional_count,
  'fractional_ratio': fractional_count/len(vars_rel)
})
```

---

# SECTION 3 — Structural properties of the constraint matrix

**Goal:** provide quick statistics about coefficient magnitudes, sparsity, and presence of big-M patterns.

**Steps for the agent:**

1. Extract coefficient statistics across nonzeros in A: min, max, mean, median, std, percentile (e.g., 90th). Save as `coeff_stats.json`.
2. Compute matrix density = nonzeros / (num_rows * num_cols).
3. Heuristic for big-M detection: search for very large coefficients relative to median/mean (e.g., any coefficient > 1e3 * median_coefficient or an absolute threshold supplied by the user). Report suspected big-M constraints.
4. Report variable and constraint counts, and average nonzeros per row and per column.

**Deliverables:**

* `coeff_stats.json` with statistics and `big_M_candidates` list.
* `report.json` additions: `num_rows`, `num_cols`, `nonzeros`, `density`, `avg_nz_per_row`, `avg_nz_per_col`.

**Example code (pseudocode outline):**

```python
# iterate over model.getCoeff(i,j) or extract sparse representation
# collect absolute values of coefficients into a list 'coefs'
import numpy as np
coefs = np.array(coefs)
stats = {
  'min': float(coefs.min()),
  'max': float(coefs.max()),
  'mean': float(coefs.mean()),
  'median': float(np.median(coefs)),
  'std': float(coefs.std()),
  '90pct': float(np.percentile(coefs,90))
}
# big-M heuristic
bigM_thresh = max(1e3*stats['median'], 1e6)  # agent may tune
big_M_candidates = find_constraints_with_coef_above(bigM_thresh)
```

---

# SECTION 4 — Symmetry & repeated blocks

**Goal:** detect obvious symmetry and repeated blocks that might cause branching redundancy.

**Steps for the agent:**

1. Look for repeated rows or columns (identical coefficient patterns). If many identical columns exist, report `num_identical_column_groups` and example variables.
2. For groups of variables with identical objective coefficients and identical constraint coefficient vectors, flag as `symmetric_group` candidates.
3. Provide a small report listing top symmetric groups (group size, representative var names).
4. Suggest quick remedies in the log (e.g., add ordering constraints, or aggregate symmetric variables where feasible).

**Deliverables:**

* `report.json` additions: `symmetric_groups` (list of dicts with `size`, `repr_vars`, `pattern_hash`), `num_symmetric_groups`.

**Example detection approach:**

* Hash normalized column vectors (e.g., sparse pattern + scaled coefficients rounded to a tolerance). Group by hash and count sizes. Report groups with size >= 2.

---

## Final instructions to agent

* Run sections sequentially; the LP relaxation from Section 1 must be reused in Section 2.
* All numeric thresholds (fractionality tolerance, big-M multiplier, hashing tolerance) should be configurable via a small top-of-file config block.
* Save intermediate artifacts in `out_dir` and produce a single consolidated `report.json` with all metrics.

---

## Minimal `report.json` schema (example)

```json
{
  "lp_obj": 123.45,
  "int_obj": 150.0,
  "integrality_gap": 0.173,
  "fractional_count": 42,
  "fractional_ratio": 0.12,
  "num_rows": 500,
  "num_cols": 200,
  "nonzeros": 4200,
  "density": 0.042,
  "coeff_stats": { "min": 0.0, "max": 1e6, "median": 2.3 },
  "big_M_candidates": ["constr_17", "constr_202"],
  "num_symmetric_groups": 3,
  "symmetric_groups": [ { "size": 10, "repr_vars": ["x_1","x_2"] } ]
}
```

---

If anything fails (solver error, infeasible LP, no incumbent), the agent should still save `report.json` and include an `errors` field describing what happened.

**End of prompt**
