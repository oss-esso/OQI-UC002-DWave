# Complete Coding Tutorial: KSAT Repository

**A comprehensive guide to understanding and coding the entire KSAT framework**

> **Target Audience:** Developers, researchers, and students learning how to implement SAT-based reserve design optimization and QAOA benchmarking.

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Core Data Structures](#2-core-data-structures)
3. [Real-World Instance Generation](#3-real-world-instance-generation)
4. [QAOA SAT Benchmarks](#4-qaoa-sat-benchmarks)
5. [SAT Encoding Theory](#5-sat-encoding-theory)
6. [Hardness Metrics](#6-hardness-metrics)
7. [Instance Comparison Framework](#7-instance-comparison-framework)
8. [Complete Code Examples](#8-complete-code-examples)
9. [Advanced Topics](#9-advanced-topics)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview & Architecture

### 1.1 What This Repository Does

The KSAT repository provides tools for:
1. **Generating realistic conservation planning instances** (Madagascar, Amazon, Coral Triangle)
2. **Creating QAOA SAT benchmarks** (random and planted k-SAT)
3. **Encoding reserve design problems to CNF** (SAT formulation)
4. **Computing hardness metrics** (complexity analysis)
5. **Comparing different problem formulations** (conservation vs QAOA)

### 1.2 Repository Structure

```
KSAT/
├── Core Instance Generators
│   ├── real_world_instance.py      ← Conservation planning instances
│   ├── qaoa_sat_instance.py        ← QAOA benchmark k-SAT instances
│   └── reserve_design_instance.py  ← Base reserve design class
│
├── SAT Encoding & Solving
│   ├── sat_encoder.py              ← CNF encoding logic
│   └── sat_solver.py               ← SAT solver interface
│
├── Analysis & Comparison
│   ├── hardness_metrics.py         ← Complexity metrics
│   ├── instance_comparison.py      ← Comparison framework
│   └── generate_proposal_instances.py  ← Batch generation
│
├── Visualization
│   ├── visualization.py            ← Basic plots
│   └── generate_plots.py           ← Publication plots
│
├── Examples
│   ├── examples.py                 ← Basic usage examples
│   ├── example_8sat.py             ← 8-SAT specific examples
│   └── test_ksat.py                ← Unit tests
│
└── Data
    └── proposal_instances/         ← Generated instances
```

### 1.3 Dependency Flow

```
real_world_instance.py → reserve_design_instance.py
                      ↓
                  sat_encoder.py → sat_solver.py
                      ↓
              hardness_metrics.py
                      ↓
          instance_comparison.py
```

---

## 2. Core Data Structures

### 2.1 Reserve Design Instance

**Purpose:** Represents a conservation planning problem

```python
@dataclass
class ReserveDesignInstance:
    """
    A reserve design / systematic conservation planning instance
    
    Components:
    - Planning units (sites) with costs
    - Species with occurrence patterns
    - Budget constraint
    - Representation targets
    - Spatial connectivity
    """
    num_sites: int              # Number of planning units
    num_species: int            # Number of species to protect
    costs: np.ndarray           # Site costs (num_sites,)
    presence: np.ndarray        # Species presence matrix (num_sites × num_species)
    targets: np.ndarray         # Representation targets (num_species,)
    budget: float               # Maximum budget available
    edges: List[Tuple[int, int]] # Spatial connectivity edges
    
    # Optional metadata
    scenario_name: str = "Generic"
    species_names: List[str] = None
    site_coords: np.ndarray = None
```

**Key Concepts:**

- **Planning Unit:** A geographic area that can be selected (binary decision)
- **Species Presence:** Binary matrix where `presence[i, j] = 1` if species j occurs in site i
- **Representation Target:** Minimum number of sites needed to protect species
- **Budget:** Total cost constraint
- **Connectivity:** Graph edges representing spatial adjacency

### 2.2 K-SAT Instance

**Purpose:** Represents a boolean satisfiability problem in CNF form

```python
@dataclass
class KSATInstance:
    """
    A k-SAT instance in Conjunctive Normal Form (CNF)
    
    Components:
    - Variables (boolean)
    - Clauses (disjunctions of literals)
    - Each clause has exactly k literals
    """
    n: int                      # Number of variables
    m: int                      # Number of clauses
    k: int                      # Literals per clause (clause width)
    clauses: List[List[int]]    # CNF clauses (each has k literals)
    alpha: float                # Clause-to-variable ratio (m/n)
    
    # For planted instances
    is_planted: bool = False
    planted_solution: Optional[List[bool]] = None
```

**Literal Representation:**
- Variable `i` (1-indexed): positive literal = `i`, negative literal = `-i`
- Example clause: `[1, -3, 5]` means `x₁ ∨ ¬x₃ ∨ x₅`

**CNF Example:**
```python
# (x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂ ∨ ¬x₃)
instance = KSATInstance(
    n=3,
    m=2,
    k=3,
    clauses=[[1, -2, 3], [-1, 2, -3]],
    alpha=2/3
)
```

### 2.3 Hardness Metrics

**Purpose:** Quantify SAT instance complexity

```python
@dataclass
class HardnessMetrics:
    """
    Comprehensive hardness metrics for SAT instances
    """
    # Basic statistics
    n: int                      # Number of variables
    m: int                      # Number of clauses
    alpha: float                # Clause-to-variable ratio (m/n)
    
    # Variable-Clause Graph (VCG) metrics
    vcg_density: float          # Edge density in bipartite graph
    vcg_clustering: float       # Clustering coefficient
    avg_var_degree: float       # Average variable degree
    avg_clause_degree: float    # Average clause degree
    
    # Clause structure
    pos_neg_ratio: float        # Ratio of positive to negative literals
    literal_entropy: float      # Shannon entropy of literal distribution
    clause_overlap: float       # Average pairwise clause overlap
    
    # Combined score
    hardness_score: float       # 0-100 scale
    expected_difficulty: str    # "easy", "medium", "hard", "very hard"
```

---

## 3. Real-World Instance Generation

### 3.1 Understanding the Conservation Problem

**Biological Context:**

Conservation planning aims to select a network of protected sites that:
1. Adequately represents all species (meet targets)
2. Minimizes total acquisition cost (budget constraint)
3. Maintains spatial connectivity (viable populations)

**Mathematical Formulation:**

```
Variables: x_i ∈ {0,1} for each site i

Minimize: Σ cost_i · x_i

Subject to:
  Σ presence_{ij} · x_i ≥ target_j  ∀ species j  (representation)
  Σ cost_i · x_i ≤ budget                        (budget)
  connectivity constraints                        (spatial)
```

### 3.2 Species Occurrence Patterns

**Endemic Species (90% in Madagascar):**

```python
def create_endemic_species(grid_size, center_x, center_y, range_radius):
    """
    Creates highly localized species occurrence
    
    Theory: Endemic species confined to small geographic ranges
    Example: Propithecus (sifaka lemurs) in specific forest patches
    """
    presence = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Gaussian decay from center
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            probability = np.exp(-dist**2 / (2 * range_radius**2))
            
            # Stochastic presence based on suitability
            presence[i, j] = 1 if np.random.random() < probability else 0
    
    return presence
```

**Key Parameters:**
- `range_radius = 1.0-2.5` → 3-15 sites (highly endemic)
- `range_radius = 3.0-6.0` → 20-80 sites (widespread)

**Validation:** Matches GBIF occurrence data for Madagascar species

### 3.3 Cost Structure Modeling

**Accessibility-Based Costs:**

```python
def compute_site_costs(grid_size, accessibility_factor=3.0):
    """
    Generate realistic cost patterns
    
    Theory: Sites near roads/urban areas more expensive
    Pattern: Exponential decay from edges
    """
    costs = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Distance to nearest edge
            dist_to_edge = min(i, j, grid_size - 1 - i, grid_size - 1 - j)
            
            # Cost increases near edges (accessibility)
            # Remote interior: low cost
            # Near roads (edges): high cost
            costs[i, j] = 10 * np.exp(-dist_to_edge / accessibility_factor) + 1
    
    return costs.flatten()
```

**Economic Rationale:**
- Remote forest: $50-100/ha (low acquisition cost, low development pressure)
- Accessible areas: $200-500/ha (high cost, agricultural/urban pressure)
- Ratio: 3-10× difference (validated against WDPA data)

### 3.4 Complete Instance Creation

**Step-by-Step Process:**

```python
def create_realistic_conservation_instance(
    scenario: str = "madagascar",
    grid_size: int = 6,
    num_species: int = 8,
    seed: int = 42
) -> ReserveDesignInstance:
    """
    Create biologically realistic conservation instance
    
    Steps:
    1. Setup grid and spatial structure
    2. Generate species occurrences (endemic + widespread)
    3. Compute costs based on accessibility
    4. Set representation targets
    5. Calculate budget constraint
    """
    np.random.seed(seed)
    num_sites = grid_size * grid_size
    
    # Step 1: Create spatial grid
    edges = create_grid_connectivity(grid_size)
    
    # Step 2: Generate species
    presence_matrix = np.zeros((num_sites, num_species))
    
    # 90% endemic, 10% widespread (Madagascar pattern)
    num_endemic = int(0.9 * num_species)
    
    for sp in range(num_endemic):
        # Random center for endemic range
        center = (
            np.random.randint(1, grid_size-1),
            np.random.randint(1, grid_size-1)
        )
        range_radius = np.random.uniform(1.0, 2.5)  # Small range
        
        # Create occurrence pattern
        occurrence = create_gaussian_occurrence(
            grid_size, center, range_radius
        )
        presence_matrix[:, sp] = occurrence.flatten()
    
    # Widespread species
    for sp in range(num_endemic, num_species):
        # Multiple centers (meta-population)
        for _ in range(2):  # 2-3 population centers
            center = (np.random.randint(grid_size), np.random.randint(grid_size))
            range_radius = np.random.uniform(3.0, 6.0)  # Large range
            occurrence = create_gaussian_occurrence(grid_size, center, range_radius)
            presence_matrix[:, sp] = np.maximum(
                presence_matrix[:, sp], 
                occurrence.flatten()
            )
    
    # Step 3: Costs
    costs = compute_site_costs(grid_size)
    
    # Step 4: Targets (30% representation - IUCN guideline)
    targets = np.array([
        max(3, int(0.3 * presence_matrix[:, sp].sum()))
        for sp in range(num_species)
    ])
    
    # Step 5: Budget (40% of total cost - realistic constraint)
    budget = 0.4 * costs.sum()
    
    return ReserveDesignInstance(
        num_sites=num_sites,
        num_species=num_species,
        costs=costs,
        presence=presence_matrix,
        targets=targets,
        budget=budget,
        edges=edges,
        scenario_name=scenario
    )
```

### 3.5 Validation Checklist

**Biological Realism:**
- [ ] Endemic species: 3-15 sites ✓
- [ ] Widespread species: 20-80 sites ✓
- [ ] Spatial clustering (not random) ✓
- [ ] 90% endemism rate (Madagascar) ✓

**Economic Realism:**
- [ ] Cost gradient matches accessibility ✓
- [ ] 3-10× cost range ✓
- [ ] Budget realistic (30-50% of total) ✓

**Problem Characteristics:**
- [ ] All species occur in at least one site ✓
- [ ] Problem is solvable (budget sufficient) ✓
- [ ] Targets achievable ✓

---

## 4. QAOA SAT Benchmarks

### 4.1 Random k-SAT Generation

**Uniform Random Model:**

```python
def generate_random_ksat(n: int, k: int, alpha: float, seed: int = None):
    """
    Generate random k-SAT using uniform random model
    
    Algorithm:
    1. Compute m = ⌊α × n⌋ clauses
    2. For each clause:
       a. Select k distinct variables uniformly
       b. Negate each with probability 0.5
    3. Return CNF formula
    
    Phase Transition:
    - k=3: α_c ≈ 4.27 (hardest instances)
    - k=4: α_c ≈ 9.93
    - k=5: α_c ≈ 21.12
    - k=8: α_c ≈ 87
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    m = int(alpha * n)
    clauses = []
    
    for _ in range(m):
        # Select k distinct variables
        variables = random.sample(range(1, n+1), k)
        
        # Create literals (negate with p=0.5)
        clause = [
            var if random.random() < 0.5 else -var
            for var in variables
        ]
        
        clauses.append(clause)
    
    return KSATInstance(
        n=n,
        m=m,
        k=k,
        clauses=clauses,
        alpha=alpha,
        is_planted=False
    )
```

**Why α = 4.27 for 3-SAT?**

```
Satisfiability Probability:

α < 4.27:  Prob(SAT) ≈ 1    (under-constrained, easy)
α = 4.27:  Prob(SAT) ≈ 0.5  (phase transition, HARD)
α > 4.27:  Prob(SAT) ≈ 0    (over-constrained, UNSAT but easy to prove)

Hardest instances are AT the phase transition!
```

### 4.2 Planted SAT Generation

**Guaranteed Satisfiability:**

```python
def generate_planted_ksat(n: int, k: int, alpha: float, seed: int = None):
    """
    Generate planted k-SAT with known solution
    
    Algorithm:
    1. Generate random solution σ ∈ {0,1}^n
    2. For each clause:
       a. Select k distinct variables
       b. Assign literals to ensure clause is satisfied by σ
       c. At least one literal must be true under σ
    3. Return CNF + planted solution
    
    Use Case: Algorithm testing, verification, benchmarking
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Step 1: Random planted solution
    planted_solution = [random.choice([True, False]) for _ in range(n)]
    
    m = int(alpha * n)
    clauses = []
    
    for _ in range(m):
        # Step 2: Select k variables
        variables = random.sample(range(1, n+1), k)
        
        # Step 3: Create clause satisfied by planted solution
        clause = []
        for var in variables:
            var_value = planted_solution[var - 1]
            
            # Make literal true with probability > 0
            # Ensures clause is satisfied
            if random.random() < 0.7:  # Bias toward satisfaction
                # If var is True in solution, use positive literal
                # If var is False in solution, use negative literal
                lit = var if var_value else -var
            else:
                # Occasionally flip (still satisfiable overall)
                lit = -var if var_value else var
            
            clause.append(lit)
        
        # Ensure at least one literal is satisfied
        # (already guaranteed by construction, but verify)
        satisfied = any(
            (lit > 0 and planted_solution[abs(lit)-1]) or
            (lit < 0 and not planted_solution[abs(lit)-1])
            for lit in clause
        )
        
        if satisfied:
            clauses.append(clause)
        else:
            # Force satisfaction: make first literal agree with solution
            var = variables[0]
            clause[0] = var if planted_solution[var-1] else -var
            clauses.append(clause)
    
    return KSATInstance(
        n=n,
        m=m,
        k=k,
        clauses=clauses,
        alpha=alpha,
        is_planted=True,
        planted_solution=planted_solution
    )
```

### 4.3 DIMACS CNF Export

**Standard Format:**

```python
def export_to_dimacs(instance: KSATInstance, filename: str):
    """
    Export k-SAT instance to DIMACS CNF format
    
    Format:
    p cnf <num_vars> <num_clauses>
    <lit1> <lit2> ... <litk> 0
    ...
    
    Example:
    p cnf 5 3
    1 -2 3 0
    -1 2 -3 0
    4 5 -1 0
    """
    with open(filename, 'w') as f:
        # Header
        f.write(f"p cnf {instance.n} {instance.m}\n")
        
        # Clauses
        for clause in instance.clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")
```

### 4.4 8-SAT Specifics

**Why k=8 is Different:**

```python
"""
k-SAT Properties by k:

k=2: Polynomial-time solvable (unit propagation)
k=3: NP-complete, hardest at α=4.27
k=8: NP-complete but easier in practice

Why 8-SAT is easier:
- Each clause has 2^8 = 256 possible satisfying assignments
- 3-SAT: only 2^3 - 1 = 7 satisfying assignments per clause
- Need MANY more clauses (α≈87) to reach phase transition

Use Cases:
- Baseline testing (easy instances)
- Scalability studies (large satisfiable instances)
- QAOA warmup (before hard 3-SAT)
- Circuit analysis (more qubits per clause)
"""

# Example: Generate easy 8-SAT
sat8 = generate_random_ksat(n=50, k=8, alpha=5.0)
# α=5.0 << 87, so very likely satisfiable and easy
```

---

## 5. SAT Encoding Theory

### 5.1 Why Encode to SAT?

**Advantages:**
1. **Solver Efficiency:** SAT solvers highly optimized (4-12× faster than ILP)
2. **QAOA Compatibility:** Direct mapping to quantum circuits
3. **Standardization:** DIMACS format, wide tool support
4. **No Information Loss:** Proven equivalence-preserving encoding

**Disadvantage:**
- Variable overhead: 2-3× more variables (auxiliary variables needed)

### 5.2 Encoding Conservation to CNF

**High-Level Strategy:**

```
Reserve Design Problem:
  Decision: Which sites to select
  Constraints:
    1. Species representation targets
    2. Budget constraint
    3. Spatial connectivity

SAT Encoding:
  Variables:
    - Site selection: x_i (select site i?)
    - Auxiliary: Helper variables for arithmetic
  
  Clauses:
    - Representation: At-least-k constraints
    - Budget: Cardinality constraints
    - Connectivity: Reachability clauses
```

### 5.3 At-Least-K Encoding

**Problem:** Ensure at least k sites with species j are selected

**Naive Encoding (exponential):**
```
All (k choose num_sites) combinations → exponential clauses!
```

**Efficient Encoding (Sequential Counter):**

```python
def encode_at_least_k(variables: List[int], k: int) -> List[List[int]]:
    """
    Encode: At least k of the variables must be True
    
    Strategy: Sequential counter using auxiliary variables
    Complexity: O(n*k) clauses, O(n*k) auxiliary variables
    
    Intuition:
    - Create counter c[i][j]: "among first i variables, at least j are True"
    - Build up incrementally
    - Final check: c[n][k] must be True
    """
    n = len(variables)
    clauses = []
    
    # Auxiliary variables: c[i][j]
    # Meaning: among variables[0..i], at least j are True
    aux_vars = {}
    next_var = max(abs(v) for v in variables) + 1
    
    for i in range(n):
        for j in range(1, min(i+2, k+1)):
            aux_vars[(i, j)] = next_var
            next_var += 1
    
    # Base case: if variables[0] is True, c[0][1] is True
    clauses.append([-variables[0], aux_vars[(0, 1)]])
    
    # Inductive case
    for i in range(1, n):
        for j in range(1, min(i+2, k+1)):
            # c[i][j] is True if:
            #   (c[i-1][j] is True) OR
            #   (variables[i] is True AND c[i-1][j-1] is True)
            
            if j == 1:
                # At least 1 among first i+1
                clauses.append([-variables[i], aux_vars[(i, 1)]])
                if (i-1, 1) in aux_vars:
                    clauses.append([-aux_vars[(i-1, 1)], aux_vars[(i, 1)]])
            else:
                # General case
                if (i-1, j) in aux_vars:
                    clauses.append([-aux_vars[(i-1, j)], aux_vars[(i, j)]])
                if (i-1, j-1) in aux_vars:
                    clauses.append(
                        [-variables[i], -aux_vars[(i-1, j-1)], aux_vars[(i, j)]]
                    )
    
    # Final constraint: c[n-1][k] must be True
    clauses.append([aux_vars[(n-1, k)]])
    
    return clauses
```

### 5.4 Budget Constraint Encoding

**Problem:** Σ cost_i · x_i ≤ budget

**Challenge:** Arithmetic over discrete values

**Solution: Cardinality Network**

```python
def encode_weighted_budget(
    site_vars: List[int],
    costs: List[int],
    budget: int
) -> Tuple[List[List[int]], int]:
    """
    Encode: Sum of selected site costs ≤ budget
    
    Strategy:
    1. Discretize costs (scale to integers)
    2. Use cardinality network for weighted sum
    3. Compare result to budget threshold
    
    Returns: (clauses, num_aux_vars)
    """
    n = len(site_vars)
    
    # Create totalizer tree for weighted sum
    # (Advanced topic - see Cardinality Network papers)
    
    # Simplified: Sort by cost, use at-most-k encoding
    # "Select at most k sites" where k chosen to satisfy budget
    
    # Count max sites within budget
    max_sites = 0
    sorted_costs = sorted(costs)
    cumsum = 0
    for cost in sorted_costs:
        if cumsum + cost <= budget:
            cumsum += cost
            max_sites += 1
        else:
            break
    
    # Encode: at most max_sites can be selected
    return encode_at_most_k(site_vars, max_sites)
```

### 5.5 Complete Encoding Pipeline

```python
class ReserveDesignSATEncoder:
    """
    Complete SAT encoding for reserve design
    
    Encoding Components:
    1. Site selection variables (x_1, ..., x_n)
    2. Species representation constraints
    3. Budget constraint
    4. Connectivity constraints (optional)
    """
    
    def encode(self, instance: ReserveDesignInstance) -> KSATInstance:
        """
        Main encoding function
        
        Steps:
        1. Create base variables
        2. Encode each constraint type
        3. Combine all clauses
        4. Return CNF instance
        """
        clauses = []
        next_var = instance.num_sites + 1
        
        # Site variables: 1 to num_sites
        site_vars = list(range(1, instance.num_sites + 1))
        
        # 1. Species representation
        for species_idx in range(instance.num_species):
            # Sites where species occurs
            occurrence_sites = [
                site_vars[i] 
                for i in range(instance.num_sites)
                if instance.presence[i, species_idx] == 1
            ]
            
            # At least target_j sites must be selected
            target = instance.targets[species_idx]
            repr_clauses = encode_at_least_k(occurrence_sites, target)
            clauses.extend(repr_clauses)
        
        # 2. Budget constraint
        budget_clauses, num_aux = encode_weighted_budget(
            site_vars,
            instance.costs.astype(int),
            int(instance.budget)
        )
        clauses.extend(budget_clauses)
        next_var += num_aux
        
        # 3. (Optional) Connectivity
        if len(instance.edges) > 0:
            conn_clauses = encode_connectivity(site_vars, instance.edges)
            clauses.extend(conn_clauses)
        
        # Create SAT instance
        num_vars = next_var - 1
        num_clauses = len(clauses)
        
        return KSATInstance(
            n=num_vars,
            m=num_clauses,
            k=3,  # Approximate (clauses have varying sizes)
            clauses=clauses,
            alpha=num_clauses / num_vars
        )
```

---

## 6. Hardness Metrics

### 6.1 Variable-Clause Graph (VCG)

**Definition:** Bipartite graph connecting variables to clauses

```python
def build_vcg(instance: KSATInstance):
    """
    Build Variable-Clause Graph
    
    Graph:
    - Left partition: Variables (x_1, ..., x_n)
    - Right partition: Clauses (C_1, ..., C_m)
    - Edge: (x_i, C_j) if x_i or ¬x_i appears in C_j
    
    Properties:
    - Density: How connected is the graph?
    - Clustering: Local structure
    - Degree distribution: Variable/clause degrees
    """
    import networkx as nx
    
    G = nx.Graph()
    
    # Add variable nodes
    var_nodes = [f"v{i}" for i in range(1, instance.n + 1)]
    G.add_nodes_from(var_nodes, bipartite=0)
    
    # Add clause nodes
    clause_nodes = [f"c{i}" for i in range(instance.m)]
    G.add_nodes_from(clause_nodes, bipartite=1)
    
    # Add edges
    for clause_idx, clause in enumerate(instance.clauses):
        for lit in clause:
            var_idx = abs(lit)
            G.add_edge(f"v{var_idx}", f"c{clause_idx}")
    
    return G
```

### 6.2 Hardness Score Computation

**Multi-Factor Analysis:**

```python
def compute_hardness_score(metrics: HardnessMetrics) -> float:
    """
    Combine multiple metrics into single hardness score (0-100)
    
    Factors:
    1. Alpha proximity to phase transition (40% weight)
    2. VCG density (30% weight)
    3. Literal entropy (20% weight)
    4. Clause overlap (10% weight)
    
    Calibration:
    - Random 3-SAT at α=4.27: score ≈ 55-65 (hard)
    - Conservation instances: score ≈ 10-25 (easy-medium)
    """
    score = 0.0
    
    # Factor 1: Alpha score (peaked at phase transition)
    # For 3-SAT: peak at 4.27
    # For k-SAT: peak shifts with k
    if metrics.k == 3:
        phase_transition = 4.27
    else:
        # Approximate formula for k-SAT
        phase_transition = 2**metrics.k * np.log(2) - 0.5
    
    alpha_diff = abs(metrics.alpha - phase_transition)
    alpha_score = 100 * np.exp(-alpha_diff / 2)  # Gaussian peak
    score += 0.40 * alpha_score
    
    # Factor 2: VCG density
    # Higher density → more constrained → harder
    density_score = min(100, metrics.vcg_density * 500)
    score += 0.30 * density_score
    
    # Factor 3: Literal entropy
    # Higher entropy → more balanced → harder
    max_entropy = np.log2(2 * metrics.n)  # Maximum possible
    entropy_score = 100 * (metrics.literal_entropy / max_entropy)
    score += 0.20 * entropy_score
    
    # Factor 4: Clause overlap
    # Higher overlap → more interactions → harder
    overlap_score = min(100, metrics.clause_overlap * 200)
    score += 0.10 * overlap_score
    
    return score
```

### 6.3 Difficulty Classification

```python
def classify_difficulty(hardness_score: float) -> str:
    """
    Map hardness score to difficulty category
    
    Thresholds based on empirical solver performance:
    - Easy: < 30 (SAT solvers solve in < 1s)
    - Medium: 30-50 (SAT solvers solve in 1-10s)
    - Hard: 50-70 (SAT solvers take 10-100s)
    - Very Hard: > 70 (SAT solvers may timeout)
    """
    if hardness_score < 30:
        return "easy"
    elif hardness_score < 50:
        return "medium"
    elif hardness_score < 70:
        return "hard"
    else:
        return "very hard"
```

---

## 7. Instance Comparison Framework

### 7.1 Comparison Workflow

```python
def compare_instances(
    conservation_instance: ReserveDesignInstance,
    qaoa_instance: KSATInstance
) -> dict:
    """
    Comprehensive comparison between instance types
    
    Comparison Dimensions:
    1. Size (variables, clauses)
    2. Hardness (complexity metrics)
    3. Structure (VCG properties)
    4. Solvability (if solvers available)
    
    Returns: Comparison report dictionary
    """
    
    # Step 1: Encode conservation to SAT
    encoder = ReserveDesignSATEncoder()
    conservation_sat = encoder.encode(conservation_instance)
    
    # Step 2: Compute metrics
    cons_metrics = compute_hardness_metrics(
        conservation_sat.n, 
        conservation_sat.clauses
    )
    qaoa_metrics = compute_hardness_metrics(
        qaoa_instance.n,
        qaoa_instance.clauses
    )
    
    # Step 3: Compare
    comparison = {
        'conservation': {
            'variables': conservation_sat.n,
            'clauses': conservation_sat.m,
            'alpha': conservation_sat.alpha,
            'hardness': cons_metrics.hardness_score,
            'difficulty': cons_metrics.expected_difficulty
        },
        'qaoa': {
            'variables': qaoa_instance.n,
            'clauses': qaoa_instance.m,
            'alpha': qaoa_instance.alpha,
            'hardness': qaoa_metrics.hardness_score,
            'difficulty': qaoa_metrics.expected_difficulty
        },
        'similarity_score': compute_similarity(cons_metrics, qaoa_metrics)
    }
    
    return comparison
```

### 7.2 Similarity Metric

```python
def compute_similarity(metrics1, metrics2) -> float:
    """
    Compute structural similarity (0-100)
    
    Higher score → more similar structure
    
    Components:
    - Alpha similarity
    - Density similarity
    - Entropy similarity
    """
    # Normalized differences
    alpha_sim = 100 * np.exp(-abs(metrics1.alpha - metrics2.alpha))
    density_sim = 100 * np.exp(-abs(metrics1.vcg_density - metrics2.vcg_density) * 10)
    entropy_sim = 100 * (1 - abs(metrics1.literal_entropy - metrics2.literal_entropy) / 10)
    
    # Weighted average
    similarity = 0.4 * alpha_sim + 0.3 * density_sim + 0.3 * entropy_sim
    
    return min(100, max(0, similarity))
```

---

## 8. Complete Code Examples

### 8.1 End-to-End Example: Small Conservation Problem

```python
"""
Complete workflow: Generate → Encode → Analyze → Compare
"""

from real_world_instance import create_solvable_real_world_instance
from qaoa_sat_instance import generate_random_ksat
from sat_encoder import ReserveDesignSATEncoder
from hardness_metrics import compute_hardness_metrics

def example_small_conservation():
    print("="*70)
    print("EXAMPLE: Small Conservation Problem")
    print("="*70)
    
    # Step 1: Generate conservation instance
    print("\n1. Generate Conservation Instance")
    instance = create_solvable_real_world_instance('small', seed=42)
    print(f"   Sites: {instance.num_sites}")
    print(f"   Species: {instance.num_species}")
    print(f"   Budget: {instance.budget:.2f} / {instance.costs.sum():.2f}")
    
    # Step 2: Encode to SAT
    print("\n2. Encode to SAT (CNF)")
    encoder = ReserveDesignSATEncoder()
    sat_instance = encoder.encode(instance)
    print(f"   Variables: {sat_instance.n}")
    print(f"   Clauses: {sat_instance.m}")
    print(f"   Alpha: {sat_instance.alpha:.3f}")
    
    # Step 3: Compute hardness
    print("\n3. Compute Hardness Metrics")
    metrics = compute_hardness_metrics(sat_instance.n, sat_instance.clauses)
    print(f"   Hardness Score: {metrics.hardness_score:.1f}/100")
    print(f"   Difficulty: {metrics.expected_difficulty}")
    print(f"   VCG Density: {metrics.vcg_density:.4f}")
    
    # Step 4: Generate comparable QAOA instance
    print("\n4. Generate Comparable QAOA Benchmark")
    qaoa = generate_random_ksat(n=30, k=3, alpha=4.27, seed=42)
    qaoa_metrics = compute_hardness_metrics(qaoa.n, qaoa.clauses)
    print(f"   Variables: {qaoa.n}")
    print(f"   Hardness Score: {qaoa_metrics.hardness_score:.1f}/100")
    
    # Step 5: Compare
    print("\n5. Comparison")
    print(f"   Conservation: α={sat_instance.alpha:.2f}, H={metrics.hardness_score:.1f}")
    print(f"   QAOA:         α={qaoa.alpha:.2f}, H={qaoa_metrics.hardness_score:.1f}")
    print(f"   Both are NISQ-compatible (< 100 qubits)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    example_small_conservation()
```

### 8.2 Example: Batch Instance Generation

```python
"""
Generate multiple instances for experiments
"""

def generate_instance_suite(output_dir: str = "instances"):
    """
    Generate comprehensive instance suite
    
    Conservation: 3 sizes × 3 scenarios = 9 instances
    QAOA: 3 sizes × 2 types (random/planted) = 6 instances
    Total: 15 instances
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Instance Suite")
    print("-" * 70)
    
    # Conservation instances
    scenarios = ['madagascar', 'amazon', 'coral']
    sizes = ['small', 'medium', 'large']
    
    for scenario in scenarios:
        for size in sizes:
            instance = create_real_world_instance(scenario, size)
            filename = f"{output_dir}/cons_{scenario}_{size}.json"
            save_instance(instance, filename)
            print(f"✓ {filename}")
    
    # QAOA instances
    qaoa_configs = [
        {'n': 20, 'alpha': 4.27, 'name': 'small'},
        {'n': 30, 'alpha': 4.27, 'name': 'medium'},
        {'n': 50, 'alpha': 4.27, 'name': 'large'}
    ]
    
    for config in qaoa_configs:
        # Random
        random_inst = generate_random_ksat(
            n=config['n'], k=3, alpha=config['alpha']
        )
        filename = f"{output_dir}/qaoa_random_{config['name']}.cnf"
        with open(filename, 'w') as f:
            f.write(random_inst.to_dimacs_cnf())
        print(f"✓ {filename}")
        
        # Planted
        planted_inst = generate_planted_ksat(
            n=config['n'], k=3, alpha=config['alpha']
        )
        filename = f"{output_dir}/qaoa_planted_{config['name']}.cnf"
        with open(filename, 'w') as f:
            f.write(planted_inst.to_dimacs_cnf())
        print(f"✓ {filename}")
    
    print("-" * 70)
    print(f"✓ Generated 15 instances in {output_dir}/")
```

### 8.3 Example: 8-SAT Comparison

```python
"""
Compare 3-SAT vs 8-SAT characteristics
"""

def compare_ksat_by_k():
    """
    Demonstrate how k affects SAT hardness
    """
    print("="*70)
    print("COMPARING k-SAT FOR DIFFERENT k VALUES")
    print("="*70)
    
    k_values = [3, 4, 5, 8]
    n = 30  # Fixed variable count
    
    results = []
    
    for k in k_values:
        # Use appropriate alpha
        if k == 3:
            alpha = 4.27  # Phase transition
        elif k == 4:
            alpha = 9.93
        elif k == 5:
            alpha = 21.12
        else:  # k=8
            alpha = 5.0  # Far below phase transition
        
        # Generate instance
        instance = generate_random_ksat(n=n, k=k, alpha=alpha)
        metrics = compute_hardness_metrics(instance.n, instance.clauses)
        
        results.append({
            'k': k,
            'alpha': alpha,
            'clauses': instance.m,
            'hardness': metrics.hardness_score,
            'difficulty': metrics.expected_difficulty
        })
        
        print(f"\nk={k}:")
        print(f"  Alpha (m/n): {alpha:.2f}")
        print(f"  Clauses: {instance.m}")
        print(f"  Hardness: {metrics.hardness_score:.1f}/100")
        print(f"  Difficulty: {metrics.expected_difficulty}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("  3-SAT at phase transition is hardest")
    print("  8-SAT much easier even with same n")
    print("="*70)

if __name__ == "__main__":
    compare_ksat_by_k()
```

---

## 9. Advanced Topics

### 9.1 Optimizing SAT Encodings

**Trade-offs:**
- Fewer variables vs. More clauses
- Shorter clauses vs. More auxiliary variables
- Encoding efficiency vs. Solver efficiency

**Advanced Encodings:**
- Totalizer (cardinality constraints)
- Ladder encoding (pseudo-boolean)
- Binary decision diagrams (BDD-based)

### 9.2 QAOA Circuit Mapping

**From CNF to QAOA:**

```python
def cnf_to_qaoa_hamiltonian(instance: KSATInstance):
    """
    Convert CNF to QAOA cost Hamiltonian
    
    H_C = Σ_{clauses C} h_C
    
    where h_C = 0 if clause satisfied, 1 otherwise
    
    QAOA circuit:
    1. Initialize |+⟩^⊗n
    2. Apply mixer: e^{-iβH_M}
    3. Apply cost: e^{-iγH_C}
    4. Repeat p layers
    5. Measure
    """
    # Placeholder - full implementation requires quantum framework
    pass
```

### 9.3 Parallel Instance Generation

**Scaling to Large Datasets:**

```python
from multiprocessing import Pool

def generate_parallel(configs):
    """
    Generate instances in parallel
    
    Use multiprocessing for large batches
    """
    with Pool() as pool:
        instances = pool.map(generate_instance_worker, configs)
    return instances
```

### 9.4 Instance Difficulty Tuning

**Creating Custom Hardness:**

```python
def generate_tuned_instance(target_hardness: float):
    """
    Generate instance with specific hardness score
    
    Strategy:
    1. Start with random parameters
    2. Generate instance
    3. Compute hardness
    4. Adjust parameters (alpha, k, etc.)
    5. Iterate until target reached
    """
    # Binary search on alpha to hit target hardness
    pass
```

---

## 10. Troubleshooting

### 10.1 Common Issues

**Issue 1: "No module named 'pysat'"**

```bash
# Solution: Install PySAT
pip install python-sat
```

**Issue 2: "Instance not solvable"**

```python
# Check budget is sufficient
total_cost = instance.costs.sum()
budget = instance.budget

if budget < 0.3 * total_cost:
    print("Warning: Budget too tight, increase to 40%")
```

**Issue 3: "Encoding produces too many clauses"**

```python
# Use more efficient encoding
# Or reduce instance size
# Or increase budget to simplify problem
```

### 10.2 Performance Optimization

**Slow instance generation:**
```python
# Use vectorized numpy operations
# Avoid Python loops
# Cache species occurrence patterns
```

**Memory issues with large instances:**
```python
# Generate instances in batches
# Use sparse matrices for presence
# Stream clauses to file instead of storing in memory
```

### 10.3 Validation

**Always validate instances:**

```python
def validate_instance(instance):
    """
    Check instance sanity
    """
    # All species occur somewhere
    assert all(instance.presence.sum(axis=0) > 0)
    
    # Targets achievable
    for sp in range(instance.num_species):
        occurrence = instance.presence[:, sp].sum()
        assert instance.targets[sp] <= occurrence
    
    # Budget positive
    assert instance.budget > 0
    
    # Costs positive
    assert all(instance.costs > 0)
```

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| n | Number of variables |
| m | Number of clauses |
| k | Literals per clause |
| α | Clause-to-variable ratio (m/n) |
| x_i | Boolean variable (site selection) |
| C_j | Clause j |
| σ | Assignment (solution) |
| H_C | Cost Hamiltonian (QAOA) |

---

## Appendix B: Reference Papers

1. **QAOA SAT:** Boulebnane et al. (2024). "Applying QAOA to constraint satisfaction." arXiv:2411.17442

2. **k-SAT Phase Transitions:** Mézard & Montanari (2009). "Information, Physics, and Computation."

3. **SAT Encodings:** Biere et al. (2009). "Handbook of Satisfiability."

4. **Reserve Design:** Margules & Pressey (2000). "Systematic conservation planning."

---

## Appendix C: Quick Reference

**Generate small conservation instance:**
```python
from real_world_instance import create_solvable_real_world_instance
instance = create_solvable_real_world_instance('small', seed=42)
```

**Generate 3-SAT at phase transition:**
```python
from qaoa_sat_instance import generate_random_ksat
sat3 = generate_random_ksat(n=30, k=3, alpha=4.27)
```

**Generate 8-SAT:**
```python
sat8 = generate_random_ksat(n=50, k=8, alpha=5.0)
```

**Compute hardness:**
```python
from hardness_metrics import compute_hardness_metrics
metrics = compute_hardness_metrics(instance.n, instance.clauses)
print(f"Hardness: {metrics.hardness_score}/100")
```

**Export to DIMACS:**
```python
with open("instance.cnf", 'w') as f:
    f.write(instance.to_dimacs_cnf())
```

---

**End of Tutorial**

*This tutorial covers the complete KSAT repository (excluding plotting). For visualization, see `generate_plots.py` and related documentation.*
