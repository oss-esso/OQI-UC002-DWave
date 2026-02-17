# Comprehensive Comparison: Full Esteso 2022 Model vs. Project Subset

**Paper:** Esteso, A., Alemany, M.M.E., Ortiz, Á., et al. (2022). *"Crop planting and harvesting planning: Conceptual framework and sustainable multi-objective optimization for plants with variable molecule concentrations and minimum time between harvests."* Applied Mathematical Modelling, 112, 136–155. doi:10.1016/j.apm.2022.07.023

**Citation key in project:** `esteso_sustainable_2023`

---

## 1. PAPER CONTEXT

The Esteso 2022 paper presents a **multi-echelon supply chain (SC) optimization model for medicinal plant (MP) production** in Basilicata, Italy. The model covers the **full lifecycle** from planting through harvesting, drying, sorting, inventory management, and transport to processing plants. The paper focuses specifically on **biennial medicinal plants** (lemon balm and sage) whose active molecule concentration depends on plant growth time between harvests.

The **project (OQI-UC002-DWave)** re-purposes the general idea of "centralized planning of sustainable food production" but applies it to a **fundamentally different and much simpler problem**: static crop-to-farm allocation for 27 food crops, optimizing nutritional and environmental scores. The project explicitly states: *"we take a smaller instance of the problem presented in [Esteso 2023]"*.

---

## 2. FULL ESTESO 2022 MODEL

### 2A. Indices and Sets

| Symbol | Description |
|--------|-------------|
| $c$ | Crop index |
| $f$ | Farm index |
| $l$ | Processing plant index |
| $p$ | Planting period index |
| $h, h'$ | Harvest period indices |
| $s$ | Harvest season index |
| $z$ | Plant growth interval ($z = h - p$ for first season, $z = h - h'$ for subsequent seasons) |
| $t$ | Time period index |
| $P_c$ | Set of planting periods for crop $c$ |
| $S_c$ | Set of harvest seasons for crop $c$ |
| $H_{cs}$ | Set of harvest periods for crop $c$ in season $s$ |

### 2B. Parameters (Full Model)

| Symbol | Description |
|--------|-------------|
| $a_f$ | Available area on farm $f$ (hectares) |
| $y^{sz}_c$ | Yield of crop $c$ after $z$ growing periods in season $s$ |
| $am^{sz}_c$ | Concentration of active molecules after $z$ growing periods in season $s$ |
| $hi_c$ | Minimum plant growth time (periods) before harvest for crop $c$ |
| $ds^{sz}_c$ | Weight loss factor from sorting (discarding non-useful plant parts) |
| $dc_f$ | Drying capacity of farm $f$'s own dryer |
| $dcc$ | Drying capacity of the central (shared) dryer |
| $ic_f$ | Storage capacity on farm $f$ |
| $mt_c$ | Minimum transport quantity for crop $c$ to be accepted |
| $pc_c$ | Planting cost per hectare for crop $c$ |
| $ch_c$ | Harvesting cost per hectare for crop $c$ |
| $cd_c$ | Drying cost per kg on farm for crop $c$ |
| $cdc_{cf}$ | Drying cost per kg at central dryer for crop $c$ from farm $f$ |
| $cs_c$ | Sorting cost per kg for crop $c$ |
| $cif_c$ | Inventory cost per period per kg on farms for crop $c$ |
| $ct_{cfl}$ | Transport cost per kg from farm $f$ to plant $l$ for crop $c$ |
| $cip_{cl}$ | Inventory cost per period per kg at processing plant $l$ for crop $c$ |
| $pm_{cl}$ | Price paid by processing plant $l$ for one kg of crop $c$ |
| $d^t_{cl}$ | Demand for dry sorted crop $c$ at processing plant $l$ in period $t$ |

### 2C. Decision Variables (Full Model) — 13 variables

| Variable | Type | Description |
|----------|------|-------------|
| $AP^p_{cf}$ | Continuous | Area planted with crop $c$ on farm $f$ during planting period $p$ |
| $AH^{shz}_{cf}$ | Continuous | Area of crop $c$ harvested on farm $f$ during period $h$ of season $s$ after $z$ growing periods |
| $QH^{shz}_{cf}$ | Continuous | Quantity of crop $c$ harvested on farm $f$ during period $h$ of season $s$ after $z$ growing periods |
| $QD^{shz}_{cf}$ | Continuous | Quantity of crop $c$ dried on farm $f$ during period $h$ of season $s$ after $z$ growing periods |
| $QCD^{shz}_{cf}$ | Continuous | Quantity of crop $c$ dried at the central dryer from farm $f$ during period $h$ of season $s$ after $z$ growing periods |
| $ID^{shzt}_{cf}$ | Continuous | Inventory of dried crop $c$ on farm $f$ during period $t$, harvested during $h$ of season $s$ after $z$ growing periods |
| $QS^{shzt}_{cf}$ | Continuous | Quantity of crop $c$ sorted on farm $f$, harvested during $h$ of season $s$ after $z$ growing periods, sorted in period $t$ |
| $IS^{shzt}_{cf}$ | Continuous | Inventory of dried sorted crop $c$ on farm $f$ during period $t$, harvested during $h$ of season $s$ after $z$ growing periods |
| $QT^{shzt}_{cfl}$ | Continuous | Quantity of dried sorted crop $c$ transported from farm $f$ to processing plant $l$ during period $t$ |
| $YT^t_{cfl}$ | **Binary** | 1 if crop $c$ is transported from farm $f$ to plant $l$ during period $t$ |
| $I^t_{cl}$ | Continuous | Inventory of dried sorted crop $c$ at processing plant $l$ during period $t$ |
| $U_f$ | Continuous | Economic unfairness measure for farm $f$ |
| $Pr_f$ | Continuous | Profit of farm $f$ |

**Scale:** For the case study (10 farms, 2 crops, 104-week horizon), the model had **292,717 constraints** and **413,844 decision variables** (412,804 continuous, 1,040 binary).

### 2D. Objective Functions (Full Model) — 3 objectives

#### Objective 1 (Environmental): Maximize Active Molecule Concentration — $Z_A$

$$\max\ Z_A = \sum_{c} \sum_{f} \sum_{l} \sum_{s \in S_c} \sum_{h \in H_{cs}} \sum_{z \geq hi_c} \sum_{t \geq h} am^{sz}_c \cdot QT^{shzt}_{cfl} \quad (1)$$

Maximizes the total concentration of active molecules in crops delivered to processors. The concentration $am^{sz}_c$ depends on plant growth time $z$.

#### Objective 2 (Economic): Minimize Supply Chain Costs — $Z_C$

$$\min\ Z_C = \sum_c \sum_f \sum_{p \in P_c} pc_c \cdot AP^p_{cf} + \sum_c \sum_f \sum_{s} \sum_{h} \sum_{z} ch_c \cdot AH^{shz}_{cf}$$
$$+ \sum_c \sum_f \sum_{s} \sum_{h} \sum_{z} cd_c \cdot QD^{shz}_{cf} + \sum_c \sum_f \sum_{s} \sum_{h} \sum_{z} cdc_{cf} \cdot QCD^{shz}_{cf}$$
$$+ \sum_c \sum_f \sum_{s} \sum_{h} \sum_{z} \sum_{t} cs_c \cdot QS^{shzt}_{cf} + \sum_c \sum_f \sum_{s} \sum_{h} \sum_{z} \sum_{t} cif_c \cdot (ID^{shzt}_{cf} + IS^{shzt}_{cf})$$
$$+ \sum_c \sum_f \sum_l \sum_{s} \sum_{h} \sum_{z} \sum_{t} ct_{cfl} \cdot QT^{shzt}_{cfl} + \sum_c \sum_l \sum_z \sum_t cip_{cl} \cdot I^{zt}_{cl} \quad (2)$$

This includes costs for: planting, harvesting, on-farm drying, central drying, sorting, on-farm inventory (dried + sorted), transport, and processor inventory.

#### Objective 3 (Social): Minimize Economic Unfairness Among Farmers — $Z_U$

$$\min\ Z_U = \sum_f U_f \quad (3)$$

where $U_f$ captures the deviation of each farm's profit-per-hectare from the average:

$$U_f \geq \frac{\sum_{f'} Pr_{f'}}{\sum_{f'} a_{f'}} - \frac{Pr_f}{a_f} \quad \forall f \quad (22)$$

$$U_f \geq \frac{Pr_f}{a_f} - \frac{\sum_{f'} Pr_{f'}}{\sum_{f'} a_{f'}} \quad \forall f \quad (23)$$

Farm profit is defined as:

$$Pr_f = \sum_c \sum_l \sum_{s} \sum_{h} \sum_{z} \sum_{t} (pm_{cl} - ct_{cfl}) \cdot QT^{shzt}_{cfl} - \text{(planting + harvesting + drying + central drying + sorting + inventory costs for farm } f\text{)} \quad (4)$$

### 2E. Constraints (Full Model) — 16 constraint sets

| # | Eq. | Constraint | Description |
|---|-----|------------|-------------|
| 1 | (5) | **Farm area limit** | $\sum_c \sum_{p \in P_c} AP^p_{cf} \leq a_f \quad \forall f$ |
| 2 | (6) | **First-season harvest = planted area** | $AP^p_{cf} = \sum_{h \in H_{cs}, h \geq p+hi_c} AH^{sh,z=h-p}_{cf} \quad \forall c,f,p \in P_c, s=1$ |
| 3 | (7) | **Consecutive-season harvest continuity** | $\sum_{z'} AH^{(s-1)hz'}_{cf} = \sum_{h' \in H_{cs}, h' \geq h+hi_c} AH^{sh',z=h'-h}_{cf} \quad \forall c,f,s>1,h \in H_{c,s-1}$ |
| 4 | (8) | **Harvest quantity = area × yield** | $QH^{shz}_{cf} = AH^{shz}_{cf} \cdot y^{sz}_c \quad \forall c,f,s,h,z$ |
| 5 | (9) | **Drying balance** | $QH^{shz}_{cf} = QD^{shz}_{cf} + QCD^{shz}_{cf} \quad \forall c,f,s,h,z$ |
| 6 | (10) | **Farm dryer capacity** | $\sum_c \sum_{s} \sum_{z} QD^{shz}_{cf} \leq dc_f \quad \forall f,h$ |
| 7 | (11) | **Central dryer capacity** | $\sum_c \sum_f \sum_{s} \sum_{z} QCD^{shz}_{cf} \leq dcc \quad \forall h$ |
| 8 | (12) | **Dried crop inventory balance (initial)** | $ID^{shz,t=h}_{cf} = QD^{shz}_{cf} + QCD^{shz}_{cf} - QS^{shz,t=h}_{cf}$ |
| 9 | (13) | **Dried crop inventory balance (subsequent)** | $ID^{shzt}_{cf} = ID^{shz,t-1}_{cf} - QS^{shzt}_{cf} \quad t > h$ |
| 10 | (14) | **Sorting capacity** | $\sum_c \sum_{s} \sum_{h \leq t} \sum_{z} QS^{shzt}_{cf} \leq sc_f \quad \forall f,t$ |
| 11 | (15) | **Sorted crop inventory balance** | $IS^{shzt}_{cf} = IS^{shz,t-1}_{cf} + QS^{shzt}_{cf} \cdot ds^{sz}_c - \sum_l QT^{shzt}_{cfl}$ |
| 12 | (16) | **Farm storage capacity** | $\sum_c \sum_{s} \sum_{h \leq t} \sum_{z} (ID^{shzt}_{cf} + IS^{shzt}_{cf}) \leq ic_f \quad \forall f,t$ |
| 13 | (17) | **Minimum transport quantity** | $\sum_{s} \sum_{h \leq t} \sum_{z} QT^{shzt}_{cfl} \geq YT^t_{cfl} \cdot mt_c \quad \forall c,f,l,t$ |
| 14 | (18) | **Transport binary linking** | $\sum_{s} \sum_{h \leq t} \sum_{z} QT^{shzt}_{cfl} \leq YT^t_{cfl} \cdot d^t_{cl} \quad \forall c,f,l,t$ |
| 15 | (19) | **Processor inventory balance** | $I^t_{cl} = I^{t-1}_{cl} + \sum_f \sum_{s} \sum_{h \leq t} \sum_{z} QT^{shzt}_{cfl} - d^t_{cl} \quad \forall c,l,t$ |
| 16 | (20) | **Variable domains** | All flow variables continuous $\geq 0$; $YT^t_{cfl}$ binary |

**Resolution methodology:** The ε-constraint method converts the 3-objective problem to a single objective (maximize $Z_A$) subject to $Z_C \leq \varepsilon_C$ and $Z_U \leq \varepsilon_U$, solved via Gurobi.

---

## 3. PROJECT SUBSET MODEL (OQI-UC002-DWave)

### 3A. Decision Variables (Project) — 3 variables

| Variable | Type | Description |
|----------|------|-------------|
| $A_{f,c}$ | Continuous | Area (hectares) assigned to food $c$ on farm $f$ |
| $Y_{f,c}$ | Binary | 1 if food $c$ is planted on farm $f$ |
| $U_c$ | Binary | 1 if food $c$ is planted on at least one farm |

### 3B. Objectives (Project) — 4 objectives collapsed into 1 weighted sum

The project uses a **single composite weighted-sum objective**:

$$\max\ Z = \frac{1}{\sum_{f} L_f} \sum_{f \in F} \sum_{c \in C} B_c \cdot A_{f,c}$$

where the **composite benefit score** $B_c$ is:

$$B_c = w_{nv} \cdot v_{nv,c} + w_{nd} \cdot v_{nd,c} - w_{ei} \cdot v_{ei,c} + w_{af} \cdot v_{af,c} + w_{su} \cdot v_{su,c}$$

The four sub-objectives (plus sustainability) are:
1. **Maximize nutritional value** ($v_{nv,c}$, weight $w_{nv} = 0.25$)
2. **Maximize nutrient density** ($v_{nd,c}$, weight $w_{nd} = 0.2$)
3. **Minimize environmental impact** ($v_{ei,c}$, weight $w_{ei} = 0.25$) — with negative sign
4. **Maximize affordability** ($v_{af,c}$, weight $w_{af} = 0.15$)
5. **Maximize sustainability** ($v_{su,c}$, weight $w_{su} = 0.15$)

### 3C. Constraints (Project) — 6 constraint families

| # | Constraint | Description |
|---|------------|-------------|
| 1 | **Land Availability** | $\sum_{c \in C} A_{f,c} \leq L_f \quad \forall f \in F$ |
| 2 | **Minimum Planting Area** | $A_{f,c} \geq A_{min,c} \cdot Y_{f,c} \quad \forall f,c$ |
| 3 | **Maximum Planting Area / Selection Linking** | $A_{f,c} \leq L_f \cdot Y_{f,c} \quad \forall f,c$ |
| 4 | **Food Group Minimum** | $\sum_{c \in G_g} U_c \geq N_{min,g} \quad \forall g$ |
| 5 | **Food Group Maximum** | $\sum_{c \in G_g} U_c \leq N_{max,g} \quad \forall g$ |
| 6 | **U-Y Linking** | $Y_{f,c} \leq U_c$ and $U_c \leq \sum_f Y_{f,c} \quad \forall c$ |

---

## 4. DETAILED MAPPING: WHAT IS KEPT vs. DROPPED

### 4A. Objectives Mapping

| # | Esteso Objective | Status | Project Equivalent |
|---|-----------------|--------|-------------------|
| 1 | $Z_A$: Maximize active molecule concentration (Environmental) | **REINTERPRETED** | Split into two new scores: "minimize environmental impact" ($v_{ei}$) and "maximize sustainability" ($v_{su}$), both as static per-crop scores rather than dynamic molecule concentrations |
| 2 | $Z_C$: Minimize supply chain costs (Economic) | **REINTERPRETED** | Replaced by "maximize affordability" ($v_{af}$), a static per-crop affordability score. No actual cost modeling |
| 3 | $Z_U$: Minimize economic unfairness among farmers (Social) | **DROPPED** | No fairness objective exists in the project |
| — | *New:* Maximize nutritional value ($v_{nv}$) | **ADDED** | Not in Esteso — project-specific objective |
| — | *New:* Maximize nutrient density ($v_{nd}$) | **ADDED** | Not in Esteso — project-specific objective |

**Key finding:** The project's 4 objectives (nutritional value, nutrient density, affordability, environmental impact) are **entirely different** from Esteso's 3 objectives (active molecule concentration, SC costs, farmer unfairness). The project takes the *general concept* of multi-objective sustainable food production planning but defines completely new objectives based on nutritional scores from GAIN datasets.

### 4B. Decision Variables Mapping

| Esteso Variable | Status | Project Equivalent |
|----------------|--------|-------------------|
| $AP^p_{cf}$ (planted area by period) | **SIMPLIFIED** | $A_{f,c}$ (area, no time dimension) |
| $AH^{shz}_{cf}$ (harvest area by season/period/growth) | **DROPPED** | — |
| $QH^{shz}_{cf}$ (harvest quantity) | **DROPPED** | — |
| $QD^{shz}_{cf}$ (on-farm drying quantity) | **DROPPED** | — |
| $QCD^{shz}_{cf}$ (central dryer quantity) | **DROPPED** | — |
| $ID^{shzt}_{cf}$ (dried crop inventory) | **DROPPED** | — |
| $QS^{shzt}_{cf}$ (sorted quantity) | **DROPPED** | — |
| $IS^{shzt}_{cf}$ (sorted inventory) | **DROPPED** | — |
| $QT^{shzt}_{cfl}$ (transport quantity) | **DROPPED** | — |
| $YT^t_{cfl}$ (transport binary) | **DROPPED** | — |
| $I^t_{cl}$ (processor inventory) | **DROPPED** | — |
| $U_f$ (unfairness) | **DROPPED** | — |
| $Pr_f$ (farm profit) | **DROPPED** | — |
| — | **ADDED** | $Y_{f,c}$ (binary crop selection per farm) |
| — | **ADDED** | $U_c$ (unique food selection across all farms) |

**Summary:** 13 Esteso variables → 3 project variables. Only $AP^p_{cf}$ survives in simplified form as $A_{f,c}$ (no temporal index). Two new binary variables ($Y_{f,c}$, $U_c$) are introduced for the MILP/QUBO formulation.

### 4C. Constraints Mapping

| # | Esteso Constraint | Status | Project Equivalent |
|---|------------------|--------|-------------------|
| (5) | Farm area limit | **KEPT (simplified)** | Land Availability: $\sum_c A_{f,c} \leq L_f$ (no planting-period dimension) |
| (6) | First-season harvest = planted area | **DROPPED** | — |
| (7) | Consecutive-season harvest continuity | **DROPPED** | — |
| (8) | Harvest quantity = area × yield | **DROPPED** | — |
| (9) | Drying balance | **DROPPED** | — |
| (10) | Farm dryer capacity | **DROPPED** | — |
| (11) | Central dryer capacity | **DROPPED** | — |
| (12) | Dried crop inventory balance (initial) | **DROPPED** | — |
| (13) | Dried crop inventory balance (subsequent) | **DROPPED** | — |
| (14) | Sorting capacity | **DROPPED** | — |
| (15) | Sorted crop inventory balance | **DROPPED** | — |
| (16) | Farm storage capacity | **DROPPED** | — |
| (17) | Minimum transport quantity | **DROPPED** | — |
| (18) | Transport binary linking | **DROPPED** | — |
| (19) | Processor inventory balance | **DROPPED** | — |
| (20) | Variable domains | **SIMPLIFIED** | Only 3 variable types instead of 13 |
| (22)–(23) | Unfairness linearization | **DROPPED** | — |
| — | — | **ADDED** | Minimum planting area: $A_{f,c} \geq A_{min,c} \cdot Y_{f,c}$ |
| — | — | **ADDED** | Selection linking: $A_{f,c} \leq L_f \cdot Y_{f,c}$ |
| — | — | **ADDED** | Food group min/max diversity: $N_{min,g} \leq \sum_{c \in G_g} U_c \leq N_{max,g}$ |
| — | — | **ADDED** | U-Y linking: $Y_{f,c} \leq U_c$, $U_c \leq \sum_f Y_{f,c}$ |

**Summary:** 16 Esteso constraint families → 1 kept (simplified), 15 dropped. 4 new constraint families added that don't exist in Esteso.

### 4D. Parameters Mapping

| Esteso Parameter | Status | Project Equivalent |
|-----------------|--------|-------------------|
| $a_f$ (farm area) | **KEPT** | $L_f$ (total land available at farm $f$) |
| $y^{sz}_c$ (yield by growth time) | **DROPPED** | — |
| $am^{sz}_c$ (molecule concentration) | **DROPPED** | — |
| $hi_c$ (min growth time) | **DROPPED** | — |
| All cost parameters ($pc_c, ch_c, cd_c, cdc_{cf}, cs_c, cif_c, ct_{cfl}, cip_{cl}$) | **DROPPED** | — |
| $pm_{cl}$ (price paid by processor) | **DROPPED** | — |
| $d^t_{cl}$ (processor demand) | **DROPPED** | — |
| $dc_f, dcc, sc_f, ic_f$ (capacity parameters) | **DROPPED** | — |
| $mt_c$ (min transport quantity) | **DROPPED** | — |
| $ds^{sz}_c$ (sorting weight loss) | **DROPPED** | — |
| — | **ADDED** | $v_{nv,c}$ (nutritional value score) |
| — | **ADDED** | $v_{nd,c}$ (nutrient density score) |
| — | **ADDED** | $v_{ei,c}$ (environmental impact score) |
| — | **ADDED** | $v_{af,c}$ (affordability score) |
| — | **ADDED** | $v_{su,c}$ (sustainability score) |
| — | **ADDED** | $A_{min,c}$ (minimum planting area per crop) |
| — | **ADDED** | $N_{min,g}, N_{max,g}$ (food group diversity bounds) |
| — | **ADDED** | $w_{nv}, w_{nd}, w_{ei}, w_{af}, w_{su}$ (objective weights) |

---

## 5. MAJOR SIMPLIFICATIONS SUMMARY

### 5.1 Temporal Dimension — ELIMINATED
Esteso models a **104-week (2-year) planning horizon** with period-by-period planting, harvesting, drying, sorting, and transport decisions. The project uses a **single static time period** — one allocation of crops to farms with no temporal dynamics.

### 5.2 Supply Chain Echelons — ELIMINATED
Esteso models **three SC echelons**: farmers → central dryer → processors (with multiple processing plant locations). The project models **only the farm level** — there is no drying, sorting, transport, storage, processing, or demand fulfillment.

### 5.3 Post-Harvest Activities — ELIMINATED
Esteso's model includes: harvesting operations with growth-time-dependent yields, on-farm drying, central drying (shared facility), sorting (with weight loss), inventory management (dried + sorted), and transport logistics. **All post-harvest activities are absent** from the project.

### 5.4 Yield and Molecule Concentration Dynamics — ELIMINATED
In Esteso, crop yield ($y^{sz}_c$) and active molecule concentration ($am^{sz}_c$) are functions of plant growth time $z$ (time between planting/consecutive harvests), creating a nonlinear temporal dependency that is a key novelty of the paper. The project uses **static per-crop scores** from external nutritional databases (GAIN), with no temporal or biological dynamics.

### 5.5 Multi-objective Method — CHANGED
Esteso uses the **ε-constraint method** (keeping one objective, constraining others) + **TOPSIS** for solution selection from a Pareto front. The project uses a **simple weighted sum** to collapse all objectives into a single scalar, losing the Pareto-optimal solution exploration.

### 5.6 Problem Domain — DIFFERENT
Esteso focuses on **medicinal plants** (lemon balm, sage) grown for active molecule extraction by a food supplement company. The project focuses on **27 diverse food crops** (fruits, vegetables, staples, pulses, animal-source foods) for **nutritional adequacy and dietary diversity** in a development context (Indonesia-derived data).

### 5.7 Crop Properties — DIFFERENT
Esteso models **biennial crops** with multiple discrete harvests within and across seasons, minimum growth times, and perishability that requires immediate drying. The project treats crops as **uniform agricultural commodities** with only static nutritional/environmental/economic scores.

### 5.8 Problem Scale — DIFFERENT
- **Esteso case study**: 10 farms, 2 crops, 104 time periods → 413,844 variables, 292,717 constraints
- **Project**: 10–1,000 farms, 27 crops, 1 time period → 297–27,027 variables, ~120–3,500 constraints

### 5.9 Fairness — DROPPED
The social equity objective (minimizing economic unfairness among farmers) is completely absent from the project.

---

## 6. NOVEL ELEMENTS IN THE PROJECT (Not from Esteso)

| Element | Description |
|---------|-------------|
| **Food group diversity constraints** | Min/max number of unique foods per food group (fruits, vegetables, staples, pulses, animal-source foods) — dietary adequacy concept not in Esteso |
| **Nutritional scoring** | NVS-based scoring from GAIN datasets — Esteso has no nutritional objectives |
| **Binary formulation variant** | Equal-plot grid formulation where $Y_{p,c} \in \{0,1\}$ replaces continuous area — designed for QUBO/quantum annealing compatibility |
| **QUBO/BQM conversion** | The entire purpose of the project is to reformulate the MILP as QUBO for quantum annealing — not relevant to Esteso |
| **Crop rotation (in formulations.tex)** | The separate MIQP formulation in `formulations.tex` adds temporal synergy (rotation) and spatial synergy between neighboring farms — these are project-specific additions inspired by agronomic principles, not from Esteso |

---

## 7. HONEST ASSESSMENT

The relationship between the Esteso 2022 paper and the project is best described as **inspirational rather than derivational**:

1. **What is borrowed from Esteso:** The *general concept* of centralized multi-objective crop planning on multiple farms, and the idea that sustainability requires balancing environmental, economic, and social dimensions. The farm area limit constraint (Eq. 5) is the only mathematical element that directly survives.

2. **What is NOT borrowed:** The entire mathematical model — all 13 decision variables (except planted area, simplified), all 3 objectives (replaced with entirely different ones), 15 of 16 constraint families, the temporal dimension, the supply chain structure, the yield dynamics, the molecule concentration model, the ε-constraint resolution, and the TOPSIS selection.

3. **The project's actual contribution:** A new, much simpler static crop allocation model designed to be amenable to QUBO reformulation and quantum annealing, applied to a broader set of food crops with nutritional/dietary objectives. The mathematical formulation is essentially original work that shares only the *framing* with Esteso.

---

*Generated: 2026-02-17*
*Sources: Esteso 2022 PDF (extracted), content_proposal.tex, full_proposal_current.tex, formulations.tex, FORMULATION_DOC_COMPARISON.md*
