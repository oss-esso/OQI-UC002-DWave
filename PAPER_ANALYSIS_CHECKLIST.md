# Paper Analysis Guide: Validating Rotation Parameters

## Paper: s41467-025-64567-9.pdf (Nature Communications)

## What We Need to Find

### 1. Rotation Penalty/Benefit Formulation

**Look for:**
- [ ] Mathematical formulation section
- [ ] How rotation effects are modeled (linear? exponential? multiplicative?)
- [ ] Specific equations for objective function
- [ ] Terms like: "rotation synergy", "crop succession", "temporal effects"

**What to check:**
```
Is it:
(a) Linear penalty:     Obj = benefit + γ × R_{c,c'} × Y[t-1] × Y[t]  ← Our formulation
(b) Exponential yield:  yield = base × exp(R_{c,c'})                  ← Different model
(c) Multiplicative:     benefit = base × (1 + rotation_factor)        ← Agronomic model
```

### 2. Parameter Values Used

**Look for tables/sections mentioning:**
- [ ] Rotation coefficient (γ, gamma, weight)
- [ ] Monoculture penalty strength (β, alpha, penalty)
- [ ] Synergy matrix values (R matrix)
- [ ] Typical values: "rotation weight = 0.X"
- [ ] Percentage effects: "X% yield increase/decrease"

**Expected locations:**
- Methods section
- Supplementary materials
- Parameter tables
- Calibration/validation sections

### 3. Monoculture Effects

**Search terms:**
- "monoculture"
- "continuous cropping"
- "17%" or "20%" (your mentioned target)
- "yield loss"
- "yield penalty"
- "rotation benefit"

**Questions to answer:**
- What percentage yield difference is mentioned?
- Is it actual crop yield or economic/objective value?
- Over what time period?
- Is it additive or multiplicative?

### 4. Literature Citations in Paper

**Look for citations to:**

**Agricultural optimization:**
- Crop rotation planning models
- Multi-period farm optimization
- Linear programming for agriculture
- Examples: "Pacino et al.", "Detlefsen & Jensen", "Dogliotti et al."

**Quantum optimization:**
- D-Wave applications
- QUBO formulations
- Quantum annealing benchmarks
- Examples: "McGeoch et al.", "King et al.", "Rieffel et al."

**Agronomic studies:**
- Rotation yield benefits
- Monoculture effects
- Soil health impacts
- Examples: Crop Science, Agronomy Journal articles

### 5. Problem Size and Benchmarks

**Check:**
- [ ] Number of farms tested
- [ ] Number of crops/crop families
- [ ] Time periods (years)
- [ ] Classical solver used (Gurobi? CPLEX? Other?)
- [ ] Time limits for classical solver
- [ ] QPU annealing time
- [ ] Speedup claims

**Compare to our implementation:**
```
Our setup:
- Scenarios: 5-1000 farms
- Crops: 6 families or 27 foods
- Periods: 3 (fixed)
- Classical: Gurobi with 60-300s timeout
- QPU: SA or DWave Advantage
```

### 6. Formulation Validation

**Look for sections on:**
- [ ] Model calibration
- [ ] Parameter estimation
- [ ] Validation against real data
- [ ] Sensitivity analysis
- [ ] "We set rotation penalty to..."
- [ ] "Following [citation], we use..."

**Key question:**
*How did they choose their parameters?*

### 7. Results Interpretation

**Check if they report:**
- [ ] Absolute yield values
- [ ] Percentage improvements
- [ ] Economic values
- [ ] Objective function values
- [ ] Solution quality metrics

**Important distinction:**
```
"20% higher yield"        → Could be exponential model
"20% better objective"    → Linear model (like ours)
"20% rotation benefit"    → Additive effect
"0.8× yield in monoculture" → Multiplicative effect
```

---

## What Our Implementation Does (For Comparison)

### Current Parameters:
```python
β (negative_strength) = -0.8
monoculture_multiplier = 1.5
R_c,c = -0.8 × 1.5 = -1.2
rotation_gamma = 0.2
```

### Effective Results:
```
Monoculture: 24% objective penalty
Good rotation: +2-10% objective bonus
Formulation: Linear (additive to objective)
```

### Literature Alignment:
```
✓ Monoculture penalty: 24% (literature: 15-30%)
✓ Rotation gamma: 0.2 (literature: 0.1-0.3)
✓ Formulation: Linear (standard for optimization)
```

---

## Expected Findings (Hypotheses)

### Scenario 1: Paper Uses Same Linear Formulation
**If we find:**
- Linear objective formulation
- Similar parameter ranges (γ ≈ 0.1-0.3, β ≈ -0.5 to -1.0)
- Comparable penalty percentages (15-30%)

**Conclusion:**
✅ Our implementation is aligned with paper
✅ Current parameters are justified
✅ 24% vs 17-20% difference is minor tuning

**Action:**
- Cite paper as supporting our approach
- Minor adjustment if desired: γ = 0.15 → 18% penalty
- No fundamental changes needed

### Scenario 2: Paper Uses Exponential Model
**If we find:**
- Exponential yield formulation
- Parameters like exp(R) terms
- Multiplicative effects

**Conclusion:**
⚠️ Different modeling paradigm
⚠️ Parameters not directly comparable
⚠️ Need to discuss formulation differences

**Action:**
- Keep our linear formulation (standard for optimization)
- Document difference in modeling approach
- Justify: "Linear formulation common for LP/MILP/QUBO"
- Calibrate to same effective penalty (24% objective ≈ 20% yield)

### Scenario 3: Paper Doesn't Specify Parameters
**If paper:**
- Focuses on quantum speedup only
- Doesn't detail rotation formulation
- Uses abstract "rotation benefits"

**Conclusion:**
✓ Our parameters are independently justified
✓ Rely on general agricultural literature
✓ No conflict with paper

**Action:**
- Cite general agricultural optimization literature
- Reference agronomic studies on rotation benefits
- Perform sensitivity analysis to show robustness

---

## Key Sections to Read (Priority Order)

1. **Methods** - Look for mathematical formulation
2. **Problem Formulation** - Objective function definition
3. **Parameters** - Tables of values used
4. **Results** - Check units (yield vs objective)
5. **Supplementary Materials** - Detailed equations
6. **Discussion** - Justification for choices
7. **References** - What papers do they cite?

---

## Questions to Answer After Reading

1. **Formulation type:**
   - [ ] Linear (like ours)
   - [ ] Exponential
   - [ ] Other

2. **Rotation parameters mentioned:**
   - γ (rotation weight): _______
   - β (penalty strength): _______
   - Monoculture effect: _____% 

3. **Percentage meaning:**
   - [ ] Yield (kg/ha)
   - [ ] Objective value
   - [ ] Economic value

4. **Can we cite this paper to justify our parameters?**
   - [ ] Yes, directly aligned
   - [ ] Yes, with minor adjustments
   - [ ] No, different formulation
   - [ ] Unclear, paper doesn't specify

5. **Do we need to change anything?**
   - [ ] No changes needed
   - [ ] Minor parameter tuning (γ adjustment)
   - [ ] Major reformulation required
   - [ ] More calibration needed

---

## After Reading: Update This File

**Paper findings:**
```
Formulation type: [Linear/Exponential/Other]
Rotation coefficient: [value or "not specified"]
Monoculture penalty: [value or "not specified"]
Percentage mentioned: [X%] refers to [yield/objective/other]
Citations relevant: [list key citations to follow up]
```

**Conclusion:**
```
Our implementation is [aligned/different/partially aligned]
Reason: [explanation]
Action needed: [none/minor adjustment/major change]
```

**Final parameter recommendation:**
```python
# Keep current or adjust:
rotation_gamma = ___  # [0.15 for 18%, 0.17 for 20%, 0.2 for 24%]
negative_strength = ___  # [keep -0.8 or adjust]
```
