# Panel Discussion: Quantum Computing for Food Production Optimization

## Prepared Answers for Q*STAR Panel Discussion

*Based on OQI Use Case Phase 3 & 4 Report: Food Production Optimization*

---

## 1. What is the single hardest computational bottleneck today in food production or plant genomics that classical methods cannot realistically overcome?

Food systems are massively complex, with dozens or hundreds of different inputs and outputs to consider. The sheer scale is the primary challenge when it comes to optimisation.

---

## 2. In your use case, what is the qualitative advantage of a quantum approach? Not speed, but what becomes possible that was not before?

there exist very specific optimization problem variants that are well-suited for a quantum annealer and where classical methods (e.g. b&b) fail, i.e. become exponentially expensive in computational resources. such problem variants cannot be solved with existing solvers (or only at the cost of a tailor-made classical solver, e.g. when Thorsten would assign a PhD student to such a problem for a couple of months, which is typically prohibitively expensive). quantum annealers can jump in, and if the hardware scales in the coming years, may be able to provide advantageous computational methods soon by solving instances of these problem variants that are intractable with existing technologies

---

## 3. Which parts of this problem space could see meaningful impact within the next 5–10 years, and which clearly remain long-term research?

Surely the applications side, implementing a solution of this kind in a real scenario to start gathering impacts

---

## 4. Agricultural data is fragmented, local, and noisy. How does this reality constrain or shape quantum-enabled approaches?

This is true in some cases, but we also have a huge amount of data at international, national, and subnational level. Food systems are well studied. There is detailed research and there are detailed data on key issues across production, consumption, and outcomes.


---

## 5. Where do you see the most critical gap today: data generation, biological modeling, computation, or integration with real-world experiments?

Based on our Phase 3/4 experience, **the most critical gap is integration with real-world experiments**—specifically, the absence of validated feedback loops between computational recommendations and field outcomes.

### Why Not the Other Gaps?

**Data generation** is active and improving: GAIN (Global Alliance for Improved Nutrition) provided our 27-crop nutritional database covering Bangladesh and Indonesia; satellite-based crop monitoring (Planet Labs, Sentinel-2) generates petabytes of agricultural imagery annually; IoT sensors increasingly track soil moisture, nutrient levels, and microclimate conditions. The data exists or is being generated—the challenge is using it effectively.


**Computation** is exactly what our project addresses. We demonstrated that D-Wave Advantage QPUs can solve rotation planning problems where Gurobi times out, with linear pure-QPU scaling and paths to further improvement through hardware topology advances. The computational gap—while not eliminated—is actively shrinking.

### The Integration Gap

What we lack is **systematic validation of quantum-optimized recommendations against agricultural outcomes**. Our 3.80× benefit improvement over Gurobi is a mathematical claim based on objective function values. Does this translate to:

- Higher actual yields in field trials?
- Reduced input costs (fertilizer, pesticides) through better rotation?
- Improved soil organic carbon over multi-year horizons?
- Acceptable implementation by farmers with real operational constraints?

We modeled rotation synergies using literature-derived parameters (16–25% legume benefits, 24% monoculture penalty), but these are population averages. The specific synergy between chickpea and subsequent wheat in a Bangladeshi farmer's field depends on that field's Rhizobium populations, drainage characteristics, and management history—variables our optimization treats as constants.


---

## 6. From your perspective, where can Japanese industry engage meaningfully at this stage? Problem definition, data sharing, field validation, or infrastructure development?

Japanese industry can engage most meaningfully at **all four stages**, but with differentiated emphasis based on Japan's distinctive strengths in the agricultural-quantum intersection.

---

## 7. What collaboration models have proven effective between frontier research platforms like OQI and industry partners, and which ones tend to fail?

### Models That Work

**1. Problem-First Partnerships**

The most effective collaborations begin with industry partners bringing *specific, well-defined problems* rather than generic "explore quantum for our domain" mandates. Our OQI use case succeeded because GAIN provided concrete nutritional datasets, EPFL contributed optimization expertise, and D-Wave offered hardware access—each partner contributed distinctly to a pre-defined problem scope (crop allocation optimization for food security).


---

## 8. Food security is both global and local. How can computational approaches developed at a global scale adapt to region-specific crops and climates, including Japan?

Our model specifically focuses on a closed ecosystem that could be a single country or a region within a country. It's a prototype that could be adapted and adopted in a very localised way.

---

## 9. Beyond yield, what metrics should define success for Zero Hunger? Resilience, sustainability, biodiversity, or something else?
Prevalence of undernourishment; proportion of people consuming all five food groups in a day; proportion of people achieving minimum dietary diversity. Alongside that, we need to provide for today without compromising tomorrow, which means reducing pressures on land and resources, destruction of biodiversity, use of fossil fuels, and overall GHG emissions from food production. It's a very multifaceted issue. 


---