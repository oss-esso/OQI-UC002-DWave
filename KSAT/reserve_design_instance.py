"""
Reserve Design Problem Instance Representation
"""

import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass, field

from KSAT.prsonal_test import compute_site_costs


@dataclass
class ReserveDesignInstance:
    """
    Reserve design problem instance
    
    Attributes:
        num_sites: Number of planning units/sites
        num_species: Number of species/features to protect
        costs: Cost of selecting each site (array of shape num_sites)
        presence: Binary matrix indicating species presence (num_sites x num_species)
        targets: Minimum representation target for each species (array of shape num_species)
        budget: Maximum total cost allowed
        adjacency: List of edges (i, j) representing adjacent sites
        max_components: Maximum number of connected components (default: 1 for full connectivity)
    """
    num_sites: int
    num_species: int
    costs: np.ndarray  # Shape: (num_sites,)
    presence: np.ndarray  # Shape: (num_sites, num_species), binary or continuous
    targets: np.ndarray  # Shape: (num_species,)
    budget: float
    adjacency: List[Tuple[int, int]] = field(default_factory=list)
    max_components: int = 1  # Default: single connected component
    site_names: Optional[List[str]] = None
    species_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate instance data"""
        assert self.costs.shape == (self.num_sites,), \
            f"Cost array shape mismatch: expected ({self.num_sites},), got {self.costs.shape}"
        assert self.presence.shape == (self.num_sites, self.num_species), \
            f"Presence matrix shape mismatch: expected ({self.num_sites}, {self.num_species}), got {self.presence.shape}"
        assert self.targets.shape == (self.num_species,), \
            f"Target array shape mismatch: expected ({self.num_species},), got {self.targets.shape}"
        assert self.budget >= 0, "Budget must be non-negative"
        assert self.max_components >= 1, "Must have at least one component"
        
        # Ensure costs are non-negative
        assert np.all(self.costs >= 0), "All costs must be non-negative"
        
        # Ensure presence is binary (0 or 1) or continuous [0, 1]
        assert np.all((self.presence >= 0) & (self.presence <= 1)), \
            "Presence values must be in [0, 1]"
        
        # Ensure targets are achievable
        for j in range(self.num_species):
            sites_with_species = np.sum(self.presence[:, j] > 0)
            assert self.targets[j] <= sites_with_species, \
                f"Target for species {j} ({self.targets[j]}) exceeds available sites ({sites_with_species})"
        
        # Set default names if not provided
        if self.site_names is None:
            self.site_names = [f"Site_{i}" for i in range(self.num_sites)]
        if self.species_names is None:
            self.species_names = [f"Species_{j}" for j in range(self.num_species)]
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Return adjacency matrix representation"""
        adj_matrix = np.zeros((self.num_sites, self.num_sites), dtype=int)
        for i, j in self.adjacency:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Undirected graph
        return adj_matrix
    
    def is_solution_feasible(self, selected_sites: List[int]) -> Tuple[bool, List[str]]:
        """
        Check if a solution is feasible
        
        Args:
            selected_sites: List of selected site indices
        
        Returns:
            (is_feasible, violations): Whether solution is feasible and list of violations
        """
        violations = []
        
        # Check budget constraint
        total_cost = sum(self.costs[i] for i in selected_sites)
        if total_cost > self.budget:
            violations.append(f"Budget exceeded: {total_cost:.2f} > {self.budget:.2f}")
        
        # Check species representation
        for j in range(self.num_species):
            count = sum(self.presence[i, j] for i in selected_sites if self.presence[i, j] > 0)
            if count < self.targets[j]:
                violations.append(
                    f"Species {self.species_names[j]} underrepresented: "
                    f"{count} < {self.targets[j]}"
                )
        
        # Check connectivity (if required)
        if self.max_components == 1 and len(selected_sites) > 1:
            if not self._is_connected(selected_sites):
                violations.append("Selected sites are not connected")
        
        return len(violations) == 0, violations
    
    def _is_connected(self, selected_sites: List[int]) -> bool:
        """Check if selected sites form a connected subgraph"""
        if len(selected_sites) <= 1:
            return True
        
        # Build adjacency list for selected sites
        site_set = set(selected_sites)
        adj_list = {site: [] for site in selected_sites}
        for i, j in self.adjacency:
            if i in site_set and j in site_set:
                adj_list[i].append(j)
                adj_list[j].append(i)
        
        # BFS from first site
        visited = set()
        queue = [selected_sites[0]]
        visited.add(selected_sites[0])
        
        while queue:
            current = queue.pop(0)
            for neighbor in adj_list[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(selected_sites)
    
    def evaluate_solution(self, selected_sites: List[int]) -> dict:
        """
        Evaluate a solution and return metrics
        
        Returns:
            Dictionary with solution metrics
        """
        total_cost = sum(self.costs[i] for i in selected_sites)
        
        species_coverage = []
        for j in range(self.num_species):
            count = sum(self.presence[i, j] for i in selected_sites if self.presence[i, j] > 0)
            species_coverage.append({
                'species': self.species_names[j],
                'target': self.targets[j],
                'achieved': count,
                'satisfied': count >= self.targets[j]
            })
        
        is_feasible, violations = self.is_solution_feasible(selected_sites)
        
        return {
            'num_selected': len(selected_sites),
            'total_cost': total_cost,
            'budget_utilization': total_cost / self.budget if self.budget > 0 else 0,
            'is_feasible': is_feasible,
            'violations': violations,
            'species_coverage': species_coverage,
            'all_species_satisfied': all(sc['satisfied'] for sc in species_coverage),
            'selected_sites': selected_sites,
            'selected_site_names': [self.site_names[i] for i in selected_sites]
        }
    
    @classmethod
    def create_random_instance(
        cls,
        num_sites: int,
        num_species: int,
        budget_fraction: float = 0.5,
        target_coverage: int = 2,
        connectivity_prob: float = 0.3,
        seed: Optional[int] = None
    ) -> 'ReserveDesignInstance':
        """
        Create a random instance for testing
        
        Args:
            num_sites: Number of sites
            num_species: Number of species
            budget_fraction: Budget as fraction of total cost
            target_coverage: Minimum sites per species
            connectivity_prob: Probability of edge between adjacent sites
            seed: Random seed for reproducibility
        
        Returns:
            Random ReserveDesignInstance
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random costs (uniform [1, 10])
        costs = np.random.uniform(1, 10, num_sites)
        
        # Random species presence (bernoulli with p=0.3)
        presence = np.random.binomial(1, 0.3, (num_sites, num_species)).astype(float)
        
        # Ensure each species appears in at least target_coverage sites
        for j in range(num_species):
            if np.sum(presence[:, j]) < target_coverage:
                # Add species to random sites
                available_sites = np.where(presence[:, j] == 0)[0]
                needed = target_coverage - int(np.sum(presence[:, j]))
                selected = np.random.choice(available_sites, size=needed, replace=False)
                presence[selected, j] = 1
        
        # Targets
        targets = np.ones(num_species) * target_coverage
        
        # Budget
        budget = budget_fraction * np.sum(costs)
        
        # Adjacency (grid-like or random)
        adjacency = []
        # Grid adjacency for spatial structure
        grid_size = int(np.ceil(np.sqrt(num_sites)))
        for i in range(num_sites):
            row, col = i // grid_size, i % grid_size
            # Right neighbor
            if col < grid_size - 1:
                j = i + 1
                if j < num_sites and np.random.random() < connectivity_prob:
                    adjacency.append((i, j))
            # Bottom neighbor
            if row < grid_size - 1:
                j = i + grid_size
                if j < num_sites and np.random.random() < connectivity_prob:
                    adjacency.append((i, j))
        
        return cls(
            num_sites=num_sites,
            num_species=num_species,
            costs=costs,
            presence=presence,
            targets=targets,
            budget=budget,
            adjacency=adjacency,
            max_components=1
        )
    
    @classmethod
    def create_grid_instance(
        cls,
        grid_rows: int,
        grid_cols: int,
        num_species: int,
        seed: Optional[int] = None
    ) -> 'ReserveDesignInstance':
        """
        Create a grid-based instance (common in spatial conservation)
        
        Args:
            grid_rows: Number of rows in grid
            grid_cols: Number of columns in grid
            num_species: Number of species
            seed: Random seed
        
        Returns:
            Grid-based ReserveDesignInstance
        """
        if seed is not None:
            np.random.seed(seed)
        
        num_sites = grid_rows * grid_cols
        
        # Costs based on position (e.g., distance from center)
        costs = np.zeros(num_sites)
        center_row, center_col = grid_rows / 2, grid_cols / 2
        for i in range(num_sites):
            row, col = i // grid_cols, i % grid_cols
            distance = np.sqrt((row - center_row)**2 + (col - center_col)**2)
            costs[i] = 1 + distance * 0.5
        
        # Species presence based on spatial patterns
        presence = np.zeros((num_sites, num_species))
        for j in range(num_species):
            # Each species has a random center and radius
            center_site = np.random.randint(num_sites)
            center_row = center_site // grid_cols
            center_col = center_site % grid_cols
            radius = np.random.uniform(2, max(grid_rows, grid_cols) / 2)
            
            for i in range(num_sites):
                row, col = i // grid_cols, i % grid_cols
                distance = np.sqrt((row - center_row)**2 + (col - center_col)**2)
                if distance <= radius:
                    presence[i, j] = 1
        
        # Targets
        targets = np.ceil(np.sum(presence, axis=0) * 0.3)  # 30% coverage
        
        # Budget (half of total cost)
        budget = np.sum(costs) * 0.5
        
        # Grid adjacency (4-connected)
        adjacency = []
        for i in range(num_sites):
            row, col = i // grid_cols, i % grid_cols
            # Right
            if col < grid_cols - 1:
                adjacency.append((i, i + 1))
            # Down
            if row < grid_rows - 1:
                adjacency.append((i, i + grid_cols))
        
        return cls(
            num_sites=num_sites,
            num_species=num_species,
            costs=costs,
            presence=presence,
            targets=targets,
            budget=budget,
            adjacency=adjacency,
            max_components=1
        )


if __name__ == "__main__":
    # Example usage
    print("Creating random instance...")
    instance = ReserveDesignInstance.create_random_instance(
        num_sites=20,
        num_species=5,
        budget_fraction=0.5,
        target_coverage=3,
        seed=42
    )
    
    print(f"Instance created:")
    print(f"  Sites: {instance.num_sites}")
    print(f"  Species: {instance.num_species}")
    print(f"  Budget: {instance.budget:.2f}")
    print(f"  Total cost: {np.sum(instance.costs):.2f}")
    print(f"  Edges: {len(instance.adjacency)}")
    
    # Test a random solution
    selected = np.random.choice(instance.num_sites, size=10, replace=False).tolist()
    evaluation = instance.evaluate_solution(selected)
    print(f"\nRandom solution evaluation:")
    print(f"  Selected: {len(selected)} sites")
    print(f"  Cost: {evaluation['total_cost']:.2f}")
    print(f"  Feasible: {evaluation['is_feasible']}")
    if not evaluation['is_feasible']:
        print(f"  Violations: {evaluation['violations']}")
    
    print("\nCreating grid instance...")
    grid_instance = ReserveDesignInstance.create_grid_instance(
        grid_rows=5,
        grid_cols=5,
        num_species=3,
        seed=42
    )
    print(f"Grid instance created: {grid_instance.num_sites} sites in 5x5 grid")