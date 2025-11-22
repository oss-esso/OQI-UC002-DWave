
import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass, field
import random

@dataclass
class ReserveDesignInstance:

    num_sites: int
    num_species: int
    costs: np.ndarray
    presence: np.ndarray

    targets: np.ndarray
    budget: float
    edges: List[Tuple[int,int]]

    scenario_name: str = 'Generic'
    species_names: List[str] = None
    site_coords: np.ndarray = None



@dataclass
class KSATInstance:

    n: int
    k: int
    m: int
    clauses: List[List[int]]
    alpha: float

    is_planted: bool = False
    planted_solution: Optional[List[bool]] = None


@dataclass
class HardnessMetrics:
    
    m: int
    n: int
    alpha: float

    vcg_density: float
    vcg_clustering: float
    avg_var_degree: float
    avg_clause_degree: float

    pos_neg_ratio: float
    literal_entropy: float
    clause_overlap: float

    hardness_score: float
    expected_diff: str


def create_endemic_species(grid_size, center_x, center_y, range_radius):
    
    presence = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            dist = np.sqrt((i-center_x)**2 + (j-center_y)**2)
            prob = np.exp(-dist**2 /(2*range_radius**2))

            presence[i,j] = 1 if np.random.random() < prob else 0


    return presence


def compute_site_costs(grid_size, accessibility_factor = 3.0):
    costs = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            dist_to_edge = min(i,j,grid_size-1-i,grid_size-1-j)

            costs[i,j] = 10*np.exp(-dist_to_edge/accessibility_factor) +1

    return costs.flatten()


def create_realistic_instance(
        scenario: str = 'madagascar',
        grid_size: int = 6,
        num_species: int = 8,
        seed: Optional[int] = 42
) -> ReserveDesignInstance:
    
    np.random.seed(seed)
    num_sites = grid_size * grid_size

    edges = create_grid_connectivity(grid_size)
    presence = np.zeros((num_sites, num_species))
    num_endemic = int(0.9 * num_species)

    for sp in range(num_endemic):

        center = (np.random.randint(1, grid_size - 1), np.random.randint(1, grid_size - 1))
        radius = np.random.uniform(1.0, 2.5)
        occurrence = create_gaussian_occurrence(grid_size, center, radius)

        presence[:, sp] = occurrence.flatten()

    for sp in range(num_endemic, num_species):

        for _ in range(2):
            center = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            radius = np.random.uniform(3.0, 6.0)
            occurrence = create_gaussian_occurrence(grid_size, center, radius)

            presence[:, sp] =np.maximum(presence[:, sp], occurrence.flatten())

    costs = compute_site_costs(grid_size)

    targets = np.array([
        max(3, int(0.3 * presence[:, sp].sum())) for sp in range(num_species)
    ])

    budget = 0.4 * costs.sum()

    return ReserveDesignInstance(
        num_sites=num_sites,
        num_species=num_species,
        costs=costs,
        presence=presence,
        targets=targets,
        budget=budget,
        edges=edges,
        scenario_name=scenario
    )


def create_grid_connectivity(grid_size: int) -> List[Tuple[int, int]]:
    edges = []
    num_sites = grid_size * grid_size

    for i in range(num_sites):
        row, col = divmod(i, grid_size)

        if col < grid_size - 1:
            edges.append((i, i + 1))

        if row < grid_size - 1:
            edges.append((i, i + grid_size))

    return edges

def create_gaussian_occurrence(grid_size: int, center: Tuple[int, int], radius: float) -> np.ndarray:
    occurrence = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            occurrence[i, j] = np.exp(-0.5 * (dist / radius)**2)

    # Normalize to [0, 1]
    occurrence = (occurrence - occurrence.min()) / (occurrence.max() - occurrence.min())

    return occurrence


def generate_random_ksat(
        n: int,
        k: int,
        alpha: float,
        seed: Optional[int] = None
) -> KSATInstance:
    
    if seed is not None:
        np.random.seed(seed)

    m = int(alpha * n)
    clauses = []

    for _ in range(m):
        clause = set()
        while len(clause) < k:
            var = np.random.randint(1, n + 1)
            sign = np.random.choice([True, False])
            literal = var if sign else -var
            clause.add(literal)
        clauses.append(list(clause))

    return KSATInstance(n=n, k=k, m=m, clauses=clauses, alpha=alpha)


def generate_planted_ksat(
        n: int,
        k: int,
        alpha: float,
        seed: Optional[int] = None
) -> KSATInstance:

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    planted_solution = [random.choice([True, False]) for _ in range(n)]

    m = int(alpha * n)
    clauses = []

    for _ in range(m):

        variables = random.sample(range(1, n + 1), k)

        clause = []
        for var in variables:
            var_value = planted_solution[var - 1]

            if random.random() < 0.7:
                literal = var if var_value else -var
            else:
                literal = -var if var_value else var

            clause.append(literal)

        satisfied = any(
            (literal > 0 and planted_solution[abs(literal)-1]) or
            (literal < 0 and not planted_solution[abs(literal)-1])
            for literal in clause
        )

        if satisfied:
            clauses.append(clause)

        else:
            var = variables[0]
            clause[0] = var if planted_solution[var - 1] else -var
            clauses.append(clause)

    return KSATInstance(
        n=n,
        k=k,
        m=m,
        clauses=clauses,
        alpha=alpha,
        is_planted=True,
        planted_solution=planted_solution
    )



def main():

    instance = create_realistic_instance(
        scenario='test_scenario',
        grid_size=6,
        num_species=10,
        seed=123
    )

    print("Reserve Design Instance:")
    print(f"Number of Sites: {instance.num_sites}")
    print(f"Number of Species: {instance.num_species}")
    print(f"Costs: {instance.costs}")
    print(f"Presence Matrix:\n{instance.presence}")
    print(f"Targets: {instance.targets}")
    print(f"Budget: {instance.budget}")
    print(f"Edges: {instance.edges}")

    ksat_instance = generate_planted_ksat(
        n=20,
        k=3,
        alpha=4.2,
        seed=42
    )

    print("\nKSAT Instance:")
    print(f"Number of Variables (n): {ksat_instance.n}")
    print(f"Clause Size (k): {ksat_instance.k}")
    print(f"Number of Clauses (m): {ksat_instance.m}")
    print(f"Clauses: {ksat_instance.clauses}")
    print(f"Is Planted: {ksat_instance.is_planted}")
    print(f"Planted Solution: {ksat_instance.planted_solution}")



if __name__ == "__main__":
    main()