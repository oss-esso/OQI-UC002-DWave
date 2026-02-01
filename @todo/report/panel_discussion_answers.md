# Panel Discussion: Quantum Computing for Food Production Optimization

## Prepared Answers for Q*STAR Panel Discussion

*Based on OQI Use Case Phase 3 & 4 Report: Food Production Optimization*

---

## 1. What is the single hardest computational bottleneck today in food production or plant genomics that classical methods cannot realistically overcome?

The single hardest computational bottleneck in food production optimization is **multi-period rotation planning with frustrated constraints at scale**. Our research identifies this as a fundamentally different challenge than simple crop allocation—one where classical mixed-integer programming (MIP) solvers consistently fail.

In our experiments, the classical state-of-the-art solver Gurobi achieved optimal solutions for single-period binary crop allocation in under 1.2 seconds for problems with up to 27,000 variables. However, when we introduced multi-period rotation planning with temporal synergies (where crop sequences matter for soil health, nitrogen fixation, and pest management) and spatial interactions (where neighboring farms influence each other ecologically), the computational landscape transformed dramatically. Gurobi hit its 300-second timeout on **11 of 13** benchmark scenarios, with MIP gaps reaching as high as 352,822%—meaning the solver could not even certify how far its solutions were from optimal.

The mathematical root of this difficulty is **frustration**—a concept from statistical physics where pairwise interactions create incompatible local optima. In crop rotation, this manifests when legume-cereal rotations are beneficial (nitrogen fixation), but monoculture is penalized, and spatial proximity creates pest transmission risks. With 70-88% of crop pair interactions being antagonistic in realistic scenarios, classical branch-and-bound algorithms cannot prune the search tree effectively because the linear programming relaxation becomes weak.

This bottleneck is not merely academic. Real-world agricultural planning for food security must simultaneously optimize across nutritional value, environmental sustainability, affordability, and temporal dynamics. The combinatorial explosion—even for 50 farms across 3 growing seasons with 6 crop families—creates optimization landscapes that classical methods navigate through enumeration, which becomes computationally prohibitive. This is the regime where quantum approaches offer genuine potential: not by being universally faster, but by providing qualitatively different optimization mechanisms for frustrated, densely-coupled problems.

---

## 2. In your use case, what is the qualitative advantage of a quantum approach? Not speed, but what becomes possible that was not before?

The qualitative advantage of quantum annealing in our crop allocation use case is **the ability to explore solution spaces that classical methods cannot reach within practical time constraints**—and surprisingly, the emergence of **agriculturally superior diverse solutions** as a natural byproduct.

Our QPU-based hierarchical decomposition achieved **3.80× higher benefit values** than Gurobi across 13 rotation scenarios. This improvement increased with scale: from 2.51× at 180 variables to 5.35× at 16,200 variables. But the deeper insight is *what* the quantum solutions looked like. Where Gurobi's optimal solutions allocated 99.6% of land to a single high-benefit crop (spinach), the quantum solutions naturally distributed plantings across all 27 crop types.

This "diversity emergence" is not a bug—it's a feature with profound practical implications:

1. **Agricultural resilience**: Monoculture solutions, while mathematically optimal for benefit maximization, create catastrophic vulnerability to crop-specific diseases, pests, or market fluctuations. The 2020 Iowa derecho destroyed 3.8 million acres of corn and soybeans precisely because of regional crop concentration.

2. **Nutritional adequacy**: Food security under SDG 2 (Zero Hunger) is not merely caloric sufficiency—it requires dietary diversity across food groups. Quantum solutions inherently satisfy this requirement.

3. **Soil health dynamics**: Crop rotation synergies (legumes fixing nitrogen for subsequent cereals, for example) require temporal diversity that quantum approaches naturally incorporate through quadratic objective terms.

The mechanism enabling this is quantum tunneling through energy barriers. Where classical branch-and-bound gets trapped in locally optimal but practically poor monoculture solutions, quantum annealing explores a broader solution manifold through quantum fluctuations. The 24% one-hot constraint violation rate we observed is actually evidence of this exploration—the QPU is finding regions of the solution space that are *infeasible* under strict mathematical constraints but *superior* for real agricultural planning.

What becomes possible is a fundamentally different optimization philosophy: solutions that are "good enough" mathematically while being "excellent" practically, discovered through mechanisms that classical enumeration cannot replicate.

---

## 3. Which parts of this problem space could see meaningful impact within the next 5–10 years, and which clearly remain long-term research?

### Near-Term Impact (5–10 Years)

**1. Regional-scale crop rotation planning (50–500 farms)**
Our results demonstrate that quantum decomposition methods already achieve practical performance for problems in this range. With pure QPU time scaling linearly at approximately 0.78 ms/variable, a 10,000-farm problem would require only ~78 seconds of quantum computation. The primary barrier is classical embedding overhead (currently 95–99% of runtime), which next-generation hardware topologies (D-Wave Zephyr with degree-20+ qubits) will substantially reduce.

**2. Hybrid quantum-classical decision support tools**
D-Wave's LeapHybridCQMSolver already achieves 0% optimality gaps with consistent 5–12 second solve times for problems up to 27,000 variables. While our analysis revealed that actual QPU contribution is <5% of total computation, this hybrid paradigm provides a practical pathway for agricultural planning software that can leverage quantum speedups as hardware improves—without requiring users to understand quantum mechanics.

**3. Multi-objective sustainability optimization**
Our formulation already incorporates nutritional value, environmental impact, affordability, and sustainability scores. As carbon credit markets mature and regenerative agriculture certification becomes mainstream, the ability to optimize across these competing objectives—where quantum methods show advantages on frustrated constraint landscapes—will become commercially valuable.

### Long-Term Research (10+ Years)

**1. Continental-scale food system optimization**
Problems with 100,000+ farms require either massively parallel QPU arrays (currently impractical) or fundamental advances in problem decomposition that preserve global constraint satisfaction across millions of subproblems. Our hierarchical methods introduce 12–32% optimality gaps that compound at extreme scales.

**2. Integration with plant genomics**
The connection between crop allocation (our focus) and genomic crop improvement remains nascent. Quantum chemistry methods for simulating protein folding or metabolic pathway optimization operate on fundamentally different problem encodings (gate-based quantum computing vs. quantum annealing). Meaningful integration requires both algorithmic and hardware maturation.

**3. Real-time adaptive replanning**
Current quantum annealing requires problem submission to cloud-based QPUs with network latency. Real-time decision support for precision agriculture (responding to daily weather, pest detection, market fluctuations) demands on-premises quantum resources or fundamentally faster response times than current systems provide.

**4. Full end-to-end food security modeling**
Our work addresses production optimization, but food security encompasses distribution logistics, storage, waste reduction, and nutritional access. Unified optimization across this supply chain remains beyond current quantum capability scales.

---

## 4. Agricultural data is fragmented, local, and noisy. How does this reality constrain or shape quantum-enabled approaches?

The fragmented, local, and noisy nature of agricultural data shapes quantum-enabled approaches in three fundamental ways—some constraining, some surprisingly enabling.

### Constraints

**1. Problem formulation requires complete parameter specification**
Our CQM/BQM formulations require explicit values for every crop benefit score, rotation synergy coefficient, and spatial interaction weight. With real agricultural data, these parameters derive from heterogeneous sources: nutritional values from USDA databases, environmental impacts from lifecycle assessments, affordability from regional market surveys. Data fragmentation means parameter uncertainty propagates directly into solution quality—quantum annealing cannot compensate for wrong inputs.

**2. Calibration to local conditions is essential**
Our rotation synergy matrix $R$ was calibrated using meta-analysis findings (16–23% yield increases for legume rotations) that represent global averages. Japanese rice-vegetable rotations, Bangladeshi aquaculture-crop integration, or Swiss alpine pasture management have locally-specific synergies that require region-specific data collection efforts. The quantum algorithm provides computational speedup, but the agriculture science must be local.

**3. Stochastic sampling compounds data noise**
Quantum annealing returns samples from a Boltzmann distribution around low-energy states, with 3–5% standard deviation in objective values across 100 samples per QPU call. When input parameters already carry 10–20% uncertainty from noisy agricultural data, distinguishing true optimization improvements from combined quantum and data noise becomes challenging.

### Opportunities

**1. Decomposition aligns with data locality**
Our farm-level decomposition strategy (27 variables per subproblem, one per farm) naturally matches the locality of agricultural data. Each farm's optimization depends primarily on its own characteristics—soil type, historical yields, climate zone—with spatial interactions handled through iterative boundary coordination. This means data collection can be incremental: add farms to the model as data becomes available, without requiring complete regional surveys before computation begins.

**2. Robust solutions through diversity**
The "diversity emergence" property of quantum solutions provides inherent robustness to parameter uncertainty. If one crop's benefit score is mis-estimated, the quantum solution's distributed allocation limits exposure compared to monoculture classical solutions. This is analogous to portfolio diversification under parameter uncertainty in finance.

**3. Adaptive refinement through iteration**
Our coordinated decomposition methods use 3–5 iterative passes with boundary biases updated each round. This iterative structure naturally accommodates sequential data updates—incorporate new soil testing results, update rotation matrices with this season's observations, rerun QPU sampling. The computational cost of incremental updates is minimal compared to full re-optimization.

### Shaping the Approach

Agricultural data realities fundamentally shape our recommendation: **start with robust problem formulations on aggregated crop families (6 categories vs. 27 individual crops) where parameter uncertainty is diluted, then disaggregate solutions using local domain expertise**. Quantum methods excel at the aggregate structural optimization; human knowledge fills the fine-grained local details.

---

## 5. Where do you see the most critical gap today: data generation, biological modeling, computation, or integration with real-world experiments?

Based on our Phase 3/4 experience, **the most critical gap is integration with real-world experiments**—specifically, the absence of validated feedback loops between computational recommendations and field outcomes.

### Why Not the Other Gaps?

**Data generation** is active and improving: GAIN (Global Alliance for Improved Nutrition) provided our 27-crop nutritional database covering Bangladesh and Indonesia; satellite-based crop monitoring (Planet Labs, Sentinel-2) generates petabytes of agricultural imagery annually; IoT sensors increasingly track soil moisture, nutrient levels, and microclimate conditions. The data exists or is being generated—the challenge is using it effectively.

**Biological modeling** has achieved remarkable sophistication: process-based crop models like APSIM, DSSAT, and WOFOST simulate plant growth, soil dynamics, and water balance at field scale; meta-analyses have quantified rotation effects across thousands of trials (the Mudare et al. 2025 synthesis we cite covers 3,663 paired observations). These models have known limitations (exponential vs. linear response functions, calibration sensitivity), but they provide workable foundations.

**Computation** is exactly what our project addresses. We demonstrated that D-Wave Advantage QPUs can solve rotation planning problems where Gurobi times out, with linear pure-QPU scaling and paths to further improvement through hardware topology advances. The computational gap—while not eliminated—is actively shrinking.

### The Integration Gap

What we lack is **systematic validation of quantum-optimized recommendations against agricultural outcomes**. Our 3.80× benefit improvement over Gurobi is a mathematical claim based on objective function values. Does this translate to:

- Higher actual yields in field trials?
- Reduced input costs (fertilizer, pesticides) through better rotation?
- Improved soil organic carbon over multi-year horizons?
- Acceptable implementation by farmers with real operational constraints?

We modeled rotation synergies using literature-derived parameters (16–25% legume benefits, 24% monoculture penalty), but these are population averages. The specific synergy between chickpea and subsequent wheat in a Bangladeshi farmer's field depends on that field's Rhizobium populations, drainage characteristics, and management history—variables our optimization treats as constants.

### Bridging the Gap

Meaningful integration requires:

1. **Pilot deployments** with farming cooperatives willing to implement quantum-recommended allocations on experimental plots alongside business-as-usual controls
2. **Multi-season monitoring** to capture rotation effects that span years (nitrogen credits, disease breaks, soil structure improvement)
3. **Farmer feedback mechanisms** to identify operational constraints (equipment availability, labor timing, market access) that mathematical models miss
4. **Iterative model refinement** where observed field outcomes update synergy matrices for subsequent optimization cycles

This is slower and messier than algorithm development—but it's the gap that determines whether quantum-optimized agriculture delivers real food security impact or remains an academic demonstration.

---

## 6. From your perspective, where can Japanese industry engage meaningfully at this stage? Problem definition, data sharing, field validation, or infrastructure development?

Japanese industry can engage most meaningfully at **all four stages**, but with differentiated emphasis based on Japan's distinctive strengths in the agricultural-quantum intersection.

### Problem Definition: Immediate High-Value Engagement

Japan's agricultural context presents unique optimization challenges that would extend our use case portfolio:

1. **Rice-based rotation systems**: Japan's rice paddy agriculture involves complex water management, winter crop rotations (wheat, soybean), and intensive greenhouse vegetable production that our current rotation model doesn't address. Japanese agricultural companies (JA cooperatives, Kubota, Yanmar) possess deep domain expertise to formulate these as optimization problems.

2. **Coastal aquaculture integration**: Nori seaweed, oyster, and fish farming co-located with rice production creates multi-dimensional optimization across marine and terrestrial systems—a problem structure where quadratic interactions (nutrient flows between systems) naturally map to quantum annealing formulations.

3. **Aging farmer workforce constraints**: Japan's agricultural labor shortage (average farmer age exceeds 67) creates operational constraints that must be encoded in optimization—when solutions must be implementable by elderly operators with limited machinery options, the problem structure changes fundamentally.

Japanese industry can bring these problem definitions to OQI partnerships, extending quantum food production optimization beyond tropical staple crops to temperate intensive agriculture systems.

### Data Sharing: Strategic Contribution

Japan possesses exceptional agricultural data infrastructure:

- **Zenno JA** national cooperative data on crop yields, input usage, and market prices across 4.7 million hectares
- **MAFF (Ministry of Agriculture)** census data with field-level production statistics
- **Satellite integration** through JAXA's agricultural monitoring programs

Data sharing agreements that provide access to these datasets—anonymized or aggregated as needed—would enable calibrating quantum optimization models for Japanese conditions. Our Bangladeshi/Indonesian crop database requires extension to Japanese cultivars, climate zones, and market structures.

### Field Validation: Critical Path Forward

As identified in Question 5, the integration gap is most critical. Japanese agricultural cooperatives offer exceptional infrastructure for systematic validation:

- **Organized cooperative structure** enables experimental designs where some members implement quantum-recommended allocations while others serve as controls
- **High data fidelity** in Japanese farming records allows precise outcome measurement
- **Multi-year commitment** cultural willingness to pursue long-term improvements aligns with the multi-season rotation validation timelines needed

A pilot program with a JA cooperative implementing QPU-optimized rotation recommendations on demonstration plots would provide the real-world validation our computational results require.

### Infrastructure Development: Long-Term Partnership

Japan is investing significantly in quantum computing infrastructure:

- **Riken-IBM Quantum Innovation Centers**
- **Quantum computing research at major corporations** (Fujitsu, NEC, Toshiba)
- **National quantum technology programs** under the Moonshot R&D initiative

Engagement could include:
- **Co-development of agricultural optimization benchmarks** for Japanese quantum hardware vendors
- **Integration of quantum optimization** into existing precision agriculture platforms (Kubota Smart Agriculture, NTT Data Farm Solutions)
- **Distributed quantum computing infrastructure** that could eventually enable on-premises QPU access for real-time agricultural decision support

The strategic opportunity is positioning Japan as a leader in **applied quantum computing for food security**—leveraging both its quantum technology investments and its world-class agricultural technology ecosystem.

---

## 7. What collaboration models have proven effective between frontier research platforms like OQI and industry partners, and which ones tend to fail?

### Models That Work

**1. Problem-First Partnerships**

The most effective collaborations begin with industry partners bringing *specific, well-defined problems* rather than generic "explore quantum for our domain" mandates. Our OQI use case succeeded because GAIN provided concrete nutritional datasets, EPFL contributed optimization expertise, and D-Wave offered hardware access—each partner contributed distinctly to a pre-defined problem scope (crop allocation optimization for food security).

What works: "We need to optimize multi-period crop rotation across 500 smallholder farms while balancing nutrition, sustainability, and affordability—can quantum methods help?"

What fails: "We're interested in quantum agriculture—let's explore possibilities."

**2. Staged Risk Reduction**

OQI's phased structure (Phase 1 scoping → Phase 2 classical simulation → Phase 3 QPU implementation → Phase 4 proof-of-concept) enables progressive de-risking where either party can assess viability before committing further resources. Our progression from Gurobi benchmarking to D-Wave hybrid solvers to pure QPU decomposition allowed course correction at each stage.

Industry partners engage more readily when commitment is incremental: initial workshops to define problem scope, followed by simulation studies demonstrating feasibility, then QPU pilots with limited scope, finally full integration planning.

**3. Complementary Capability Models**

Successful collaborations leverage distinct competencies:
- **Research platforms** (OQI, universities): Algorithm development, theoretical analysis, publication preparation
- **Hardware vendors** (D-Wave, IBM): QPU access, technical support, roadmap insights
- **Industry partners**: Domain expertise, data access, deployment infrastructure, market understanding

Our project benefited from this separation: EPFL handled formulation and benchmarking; D-Wave provided Leap platform access; GAIN contributed agricultural domain expertise. No single partner could have executed the full scope.

**4. Open Intermediate Results with Protected Applications**

Publishing Phase 3 results (timing breakdowns, optimality gaps, decomposition methods) advances the field and attracts further collaboration, while specific deployment arrangements with industry partners can remain proprietary. This balances academic incentives for openness with commercial interests in competitive advantage.

### Models That Fail

**1. Technology-Push Approaches**

Collaborations initiated by quantum vendors seeking agricultural "use cases" for marketing purposes typically fail to produce deployable solutions. The problem must be authentic—something industry partners actually need to solve—rather than reverse-engineered to demonstrate quantum capability.

**2. Unrealistic Timeline Expectations**

Industry partners expecting production-ready quantum solutions within 12-month project cycles underestimate the maturation pathway. Our Phase 3 work required 18+ months to progress from problem formulation to QPU benchmarking—and real deployment would require additional validation time. Partnerships fail when commercial pressure demands premature productization.

**3. Insufficient Domain Integration**

Quantum computing researchers optimizing agricultural problems without agricultural scientists produce mathematically interesting but practically irrelevant solutions. Our rotation synergy matrices required extensive literature review and parameter calibration—shortcuts in domain integration create models that optimize the wrong objectives.

**4. One-Way Data Sharing**

Collaborations where industry provides data but receives only academic publications (not actionable tools or competitive advantage) deteriorate over time. Industry engagement requires tangible value return—whether early access to optimization methods, co-authorship, or deployment support.

**5. Scale Mismatch**

Partnerships between large multinationals and small research teams often fail due to decision-making speed differentials and resource asymmetries. Medium-sized agricultural technology companies or regional cooperatives typically offer better alignment with research group capacities.

---

## 8. Food security is both global and local. How can computational approaches developed at a global scale adapt to region-specific crops and climates, including Japan?

Our quantum optimization framework is explicitly designed for hierarchical adaptation from global methodology to local deployment through three mechanisms: modular problem formulation, parameterized synergy matrices, and decomposition strategies that match natural data locality.

### Modular Problem Formulation

The mathematical structure of our CQM/BQM formulations separates *optimization methodology* from *agricultural parameters*:

**Global (fixed) components:**
- Objective function structure: benefit maximization with penalty terms
- Constraint types: one-crop-per-slot, diversity requirements, rotation rules
- Decomposition algorithms: PlotBased, Multilevel, Coordinated methods
- QPU interface: D-Wave embedding, sampling, postprocessing

**Local (configurable) components:**
- Crop set $\mathcal{C}$: Japanese rice, wheat, soybean, vegetables vs. Bangladeshi staples
- Benefit scores $B_c$: local nutritional priorities, market prices, sustainability metrics
- Rotation matrix $R$: region-specific synergies (rice-wheat-soybean vs. corn-soybean)
- Spatial neighbor graph $\mathcal{E}$: field topology, irrigation networks, microclimate zones
- Constraint parameters: food group requirements reflecting local dietary patterns

To adapt our framework for Japan:

1. **Crop database extension**: Replace 27 tropical crops with Japanese agricultural portfolio (rice varieties, temperate vegetables, fruit orchards, greenhouse crops). Compute benefit scores using Japanese nutritional guidelines, JA market price data, and MAFF sustainability assessments.

2. **Rotation synergy calibration**: Quantify rice-wheat rotation effects from Japanese agricultural research (Tsukuba NARO datasets), greenhouse vegetable successions, and cover crop impacts. The mathematical structure ($R_{c_1,c_2}$ coefficients) remains identical; only numerical values change.

3. **Climate zone stratification**: Japan's north-south climate gradient (Hokkaido vs. Okinawa) requires either separate optimization runs per zone or climate-zone-specific constraint parameters within a unified model.

### Parameterized Synergy Matrices

Our rotation and spatial interaction matrices are fully parameterized:

$$R_{c_1,c_2} = \begin{cases} -\beta \cdot 1.5 & \text{monoculture penalty} \\ \text{Unif}(\beta \cdot 1.2, \beta \cdot 0.3) & \text{frustration interactions} \\ \text{Unif}(0.02, 0.20) & \text{beneficial rotations} \end{cases}$$

For Japan, agricultural scientists would replace these distributions with empirically-measured values:
- Monoculture penalty for continuous rice: well-documented at 10-15% yield decline
- Soybean nitrogen credit for subsequent rice: quantified in Tohoku trials at 20-40 kg N/ha equivalent
- Greenhouse heat accumulation effects on sequential crops: tomato → cucumber vs. cucumber → tomato asymmetries

The quantum annealing methodology is agnostic to whether $R_{c_1,c_2} = 0.15$ represents corn-soybean synergy in Iowa or rice-barley synergy in Niigata—it optimizes over whatever matrix is provided.

### Decomposition Strategies Match Data Locality

Our farm-level decomposition (27 variables per farm) naturally accommodates Japan's fragmented agricultural landscape:

- Average Japanese farm size: 3.0 hectares (much smaller than US/European averages)
- Field fragmentation: single farms often comprise multiple non-contiguous parcels
- Cooperative structure: JA groups aggregate small farms for collective planning

The decomposition approach—solving each farm independently then coordinating through boundary biases—maps directly to Japanese reality where each small farm has distinct local conditions but operates within cooperative frameworks. The algorithm's iterative coordination mimics how JA cooperatives already aggregate individual farm decisions into regional plans.

### Practical Adaptation Pathway

For Japanese deployment:

1. **Phase A**: Partner with agricultural research institution (NARO, prefectural experiment stations) to compile Japan-specific crop database and rotation synergy parameters
2. **Phase B**: Pilot in single JA cooperative (perhaps 50-200 farms) using their existing data infrastructure
3. **Phase C**: Validate computational recommendations against field outcomes over 2-3 growing seasons
4. **Phase D**: Scale to regional deployment with climate zone differentiation

The computational framework requires **zero** changes; only the agricultural parameterization adapts to Japanese conditions.

---

## 9. Beyond yield, what metrics should define success for Zero Hunger? Resilience, sustainability, biodiversity, or something else?

Our crop allocation optimization framework already incorporates metrics beyond yield, and our experience implementing multi-objective optimization provides perspective on which metrics prove most tractable for computational approaches and most meaningful for food security outcomes.

### Metrics We Currently Optimize

Our composite benefit score integrates five weighted components:

$$B_c = w_{nv} \cdot v_{nv,c} + w_{nd} \cdot v_{nd,c} - w_{ei} \cdot v_{ei,c} + w_{af} \cdot v_{af,c} + w_{su} \cdot v_{su,c}$$

- **Nutritional value** ($v_{nv}$): vitamins, minerals, fiber content—directly addresses hunger as malnutrition, not just caloric insufficiency
- **Nutrient density** ($v_{nd}$): protein content, calories per hectare—links to production efficiency
- **Environmental impact** ($v_{ei}$, negatively weighted): carbon footprint, water usage, land acidification—sustainability metrics
- **Affordability** ($v_{af}$): inverse cost per kg—ensures solutions are economically accessible to food-insecure populations
- **Sustainability score** ($v_{su}$): soil health impact, biodiversity support—long-term productive capacity

The flexibility of this formulation allows stakeholder-specific weight adjustment: food security programs might emphasize nutritional value and affordability; climate adaptation initiatives might weight environmental impact more heavily; regenerative agriculture advocates might prioritize sustainability scores.

### Recommended Metrics Framework for Zero Hunger

Based on our optimization experience and SDG 2 targets, I recommend a hierarchical metric framework:

**Tier 1: Essential (must-satisfy constraints)**
1. **Caloric sufficiency**: Minimum kilocalories per capita from regional production
2. **Protein adequacy**: Essential amino acid coverage for population health
3. **Micronutrient access**: Iron, zinc, vitamin A availability across food groups

**Tier 2: Resilience (objective function components)**
4. **Production diversity**: Entropy of crop distribution—our quantum solutions naturally maximize this
5. **Climate variance buffering**: Portfolio of crops with uncorrelated climate sensitivities
6. **Market shock absorption**: Diversification across crops with different market dynamics
7. **Temporal yield stability**: Variance reduction through rotation planning

**Tier 3: Sustainability (long-term constraints)**
8. **Soil carbon trajectory**: Net carbon sequestration vs. emission from agricultural land
9. **Water footprint per nutrition unit**: Efficiency of water use for nutritional output
10. **Pollinator dependency risk**: Exposure to pollination service disruptions
11. **Agrochemical input intensity**: Fertilizer and pesticide load per hectare

**Tier 4: Equity (distributional requirements)**
12. **Smallholder inclusion**: Solutions implementable on <2 hectare farms (84% of global holdings)
13. **Geographic access**: Nutritional diversity within 50km of all population centers
14. **Price stability**: Coefficient of variation in food costs for low-income consumers

### Computational Tractability

Not all metrics optimize equally well. Our experience suggests:

**Highly tractable**: Diversity (naturally emerges from quantum sampling), environmental impact (linear in crop selection), affordability (linear coefficient on benefit scores)

**Moderately tractable**: Resilience metrics requiring correlation structures (can be encoded as quadratic interactions but increase problem complexity)

**Challenging**: Temporal dynamics beyond 3 periods (problem size grows linearly with horizon), equity distributions (require disaggregated population models beyond farm-level optimization)

### The Overlooked Metric: Implementation Feasibility

Perhaps the most important metric missing from theoretical frameworks is **implementation feasibility**—whether computed solutions can actually be adopted by farmers. Our quantum solutions show higher objective values but include 24% constraint violations representing practical impossibilities (farms assigned no crops, crops exceeding available seed supply).

Success for Zero Hunger ultimately requires metrics that bridge computational optimality and operational reality: solutions that farmers with existing equipment, knowledge, and market access can implement. This argues for optimization frameworks that include **operationalization constraints**—encoding what farmers can actually do, not just what would be mathematically optimal if all actions were possible.

---

## References

1. D-Wave Systems Inc. (2020). D-Wave Advantage: The next generation of quantum computing.
2. Mudare et al. (2025). Crop rotations increase yields: A global meta-analysis. *Field Crops Research*.
3. Preissel et al. (2015). Magnitude and farm-economic value of grain legume pre-crop benefits in Europe.
4. Kirkegaard & Ryan (2014). Magnitude and mechanisms of persistent crop sequence effects on wheat.
5. Lucas (2014). Ising formulations of many NP problems. *Frontiers in Physics*.
6. FAO (2014). The State of Food and Agriculture: Innovation in family farming.
7. Lowder et al. (2021). Which farms feed the world and has farmland become more concentrated?
8. D-Wave documentation on Pegasus topology and LeapHybridCQMSolver properties.

---

*Document prepared for Q*STAR Panel Discussion*
*OQI Food Production Optimization Use Case, Phase 3/4*
*Contact: Edoardo Spigarolo, EPFL*
