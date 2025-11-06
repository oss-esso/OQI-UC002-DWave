# CQM Functions Comparison Table

## Quick Reference

| Function | Formulation | Rotation | Variables | Use Case |
|----------|-------------|----------|-----------|----------|
| `create_cqm_farm()` | Continuous (Farm) | ❌ No | A_{f,c}, Y_{f,c} | Flexible area allocation per farm |
| `create_cqm_farm_rotation_3period()` | Continuous (Farm) | ✅ 3-period | A_{f,c,t}, Y_{f,c,t} | Farm rotation with flexible areas |
| `create_cqm_plots()` | Binary (Plots) | ❌ No | Y_{p,c} | Discrete plot assignments |
| `create_cqm_plots_rotation_3period()` | Binary (Plots) | ✅ 3-period | Y_{p,c,t} | Plot rotation with discrete assignments |

## Detailed Comparison

### Without Rotation (Baseline)

#### Farm Formulation
```python
create_cqm_farm(farms, foods, food_groups, config)
```
- **Variables:** Continuous A_{f,c} ∈ [0, land_f], Binary Y_{f,c}
- **Objective:** sum_{f,c} B_c * A_{f,c}
- **Constraints:** Per-farm land limits + linking constraints

#### Plots Formulation  
```python
create_cqm_plots(farms, foods, food_groups, config)
```
- **Variables:** Binary Y_{p,c} ∈ {0,1}
- **Objective:** sum_{p,c} B_c * area_p * Y_{p,c}
- **Constraints:** One crop per plot + min/max plots per crop

---

### With 3-Period Rotation

#### Farm Formulation
```python
create_cqm_farm_rotation_3period(farms, foods, food_groups, config, gamma=0.1)
```
- **Variables:** Time-indexed A_{f,c,t}, Y_{f,c,t} for t ∈ {1,2,3}
- **Objective:** 
  - Linear: sum_{f,c,t} B_c * A_{f,c,t}
  - Quadratic: sum_{f,c,c',t} gamma * R_{c,c'} * A_{f,c,t-1} * A_{f,c',t}
- **Constraints:** Per-period land limits + per-period linking

#### Plots Formulation
```python
create_cqm_plots_rotation_3period(farms, foods, food_groups, config, gamma=0.1)
```
- **Variables:** Time-indexed Y_{p,c,t} ∈ {0,1} for t ∈ {1,2,3}
- **Objective:**
  - Linear: sum_{p,c,t} B_c * area_p * Y_{p,c,t}
  - Quadratic: sum_{p,c,c',t} gamma * area_p * R_{c,c'} * Y_{p,c,t-1} * Y_{p,c',t}
- **Constraints:** Per-period plot assignments + per-period min/max plots

---

## Key Differences

### Farm vs Plots

| Aspect | Farm (Continuous) | Plots (Binary) |
|--------|-------------------|----------------|
| **Land Model** | Variable area allocation | Fixed-size discrete units |
| **Variables** | Real + Binary | Binary only |
| **Flexibility** | Can split farm into fractions | All-or-nothing assignments |
| **Complexity** | Higher (more variables) | Lower (fewer variables) |
| **Realism** | Good for large farms | Good for uniform grids |

### Without vs With Rotation

| Aspect | Without Rotation | With Rotation |
|--------|------------------|---------------|
| **Time Horizon** | Single period | 3 periods |
| **Objective** | Linear only | Linear + Quadratic |
| **Variables** | 1× | 3× (one per period) |
| **Constraints** | Single-period | Per-period |
| **Optimization** | Immediate value | Value + rotation synergy |
| **Complexity** | Lower | Higher (3× problem size) |

---

## Decision Guide

### Choose Farm Formulation When:
- Farms have different sizes
- Need flexible area allocation
- Can split crops across farm areas
- Have realistic land distributions

### Choose Plots Formulation When:
- Land is divided into uniform units
- Each unit gets exactly one crop
- Want simpler binary problem
- Have grid-based land representation

### Add Rotation When:
- Need multi-year planning
- Want to capture rotation benefits
- Have rotation synergy data
- Can handle 3× larger problem

### Skip Rotation When:
- Single-season planning
- Problem already complex
- No rotation data available
- Need fastest solve times

---

## Performance Considerations

### Problem Size Scaling

**Without Rotation:**
- Farm: O(n_farms × n_crops) continuous + binary variables
- Plots: O(n_plots × n_crops) binary variables

**With Rotation:**
- Farm: O(3 × n_farms × n_crops) continuous + binary variables
- Plots: O(3 × n_plots × n_crops) binary variables

### Quadratic Terms

Rotation adds O(n_units × n_crops²) quadratic terms, which:
- Increase problem complexity significantly
- Require quantum/hybrid solvers (can't use simple LP)
- Capture important biological crop interactions
- May improve solution quality despite computational cost

---

## Example: When to Use Each

1. **Small-scale farm with flexible planting:** `create_cqm_farm()`
2. **Multi-year farm planning with rotation:** `create_cqm_farm_rotation_3period()`
3. **Grid-based community garden:** `create_cqm_plots()`
4. **Grid-based multi-year crop planning:** `create_cqm_plots_rotation_3period()`

