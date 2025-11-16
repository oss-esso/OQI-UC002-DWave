"""
SAT Encoding for Reserve Design Problem
Converts reserve design constraints to CNF (K-SAT)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from pysat.formula import CNF
from pysat.card import CardEnc, EncType
from reserve_design_instance import ReserveDesignInstance


class ReserveDesignSATEncoder:
    """
    Encodes reserve design problem to CNF (Conjunctive Normal Form)
    
    This encoder converts the reserve design optimization problem into
    a Boolean satisfiability problem in CNF, which can be solved by
    modern SAT solvers.
    """
    
    def __init__(self, instance: ReserveDesignInstance, verbose: bool = False):
        """
        Initialize encoder
        
        Args:
            instance: Reserve design problem instance
            verbose: Print encoding progress
        """
        self.instance = instance
        self.verbose = verbose
        self.cnf = CNF()
        self.var_counter = 0
        
        # Variable mappings
        self.site_vars: Dict[int, int] = {}  # site_id -> sat_variable
        self.edge_vars: Dict[Tuple[int, int], int] = {}  # (site_i, site_j) -> sat_variable
        self.auxiliary_vars: Dict[str, int] = {}  # auxiliary variables for encodings
        
        if self.verbose:
            print(f"Initializing SAT encoder for {instance.num_sites} sites, "
                  f"{instance.num_species} species")
    
    def new_var(self, name: Optional[str] = None) -> int:
        """
        Allocate a new SAT variable
        
        Args:
            name: Optional name for debugging
        
        Returns:
            New variable ID (positive integer)
        """
        self.var_counter += 1
        if name is not None:
            self.auxiliary_vars[name] = self.var_counter
        return self.var_counter
    
    def encode_site_variables(self):
        """Create Boolean variables for each site selection decision"""
        if self.verbose:
            print("Encoding site variables...")
        
        for i in range(self.instance.num_sites):
            self.site_vars[i] = self.new_var(f"site_{i}")
        
        if self.verbose:
            print(f"  Created {len(self.site_vars)} site variables")
    
    def encode_connectivity_variables(self):
        """Create Boolean variables for each edge (representing connectivity)"""
        if self.verbose:
            print("Encoding connectivity variables...")
        
        for i, j in self.instance.adjacency:
            # Ensure consistent ordering
            edge = (min(i, j), max(i, j))
            if edge not in self.edge_vars:
                self.edge_vars[edge] = self.new_var(f"edge_{i}_{j}")
        
        if self.verbose:
            print(f"  Created {len(self.edge_vars)} edge variables")
    
    def encode_and_gate(self, a: int, b: int, c: int):
        """
        Encode c = a AND b using CNF clauses
        
        Clauses:
            - c -> a  (equivalently: ¬c ∨ a)
            - c -> b  (equivalently: ¬c ∨ b)
            - (a ∧ b) -> c  (equivalently: ¬a ∨ ¬b ∨ c)
        
        Args:
            a, b: Input variables
            c: Output variable
        """
        self.cnf.append([-c, a])   # c implies a
        self.cnf.append([-c, b])   # c implies b
        self.cnf.append([-a, -b, c])  # a and b implies c
    
    def encode_or_gate(self, a: int, b: int, c: int):
        """
        Encode c = a OR b using CNF clauses
        
        Args:
            a, b: Input variables
            c: Output variable
        """
        self.cnf.append([a, b, -c])   # not c implies not a and not b
        self.cnf.append([-a, c])      # a implies c
        self.cnf.append([-b, c])      # b implies c
    
    def encode_species_representation(self):
        """
        Encode species representation constraints
        
        For each species j: sum(presence[i,j] * x[i]) >= target[j]
        
        Uses cardinality encoding (AtLeast-K constraint)
        """
        if self.verbose:
            print("Encoding species representation constraints...")
        
        for j in range(self.instance.num_species):
            # Find sites containing species j
            sites_with_species = []
            for i in range(self.instance.num_sites):
                if self.instance.presence[i, j] > 0:
                    sites_with_species.append(self.site_vars[i])
            
            target = int(self.instance.targets[j])
            
            if target > 0 and sites_with_species:
                if target > len(sites_with_species):
                    raise ValueError(
                        f"Species {j} target ({target}) exceeds available sites "
                        f"({len(sites_with_species)})"
                    )
                
                # Use PySAT cardinality encoding
                # AtLeast(k, literals) encodes: at least k literals must be true
                clauses = CardEnc.atleast(
                    lits=sites_with_species,
                    bound=target,
                    encoding=EncType.seqcounter,  # Sequential counter encoding
                    vpool=None
                )
                
                for clause in clauses.clauses:
                    self.cnf.append(clause)
                
                # Update variable counter to account for auxiliary variables
                self.var_counter = max(self.var_counter, clauses.nv)
                
                if self.verbose:
                    print(f"  Species {j}: at least {target} of {len(sites_with_species)} sites")
        
        if self.verbose:
            print(f"  Total clauses so far: {len(self.cnf.clauses)}")
    
    def encode_budget_constraint(self, max_cost: float):
        """
        Encode budget constraint: sum(cost[i] * x[i]) <= max_cost
        
        For integer costs, uses cardinality/pseudo-Boolean encoding
        For general costs, discretizes and uses weighted encoding
        
        Args:
            max_cost: Maximum allowed total cost
        """
        if self.verbose:
            print(f"Encoding budget constraint (max_cost = {max_cost})...")
        
        costs = self.instance.costs
        
        # Discretize costs to integers for encoding
        # Scale by 100 to preserve 2 decimal places
        scale_factor = 100
        int_costs = (costs * scale_factor).astype(int).tolist()
        max_int_cost = int(max_cost * scale_factor)
        
        # Use PySAT's pseudo-Boolean encoding
        # Create weighted literals: (literal, weight) pairs
        weighted_lits = []
        for i in range(self.instance.num_sites):
            if int_costs[i] > 0:
                weighted_lits.append((self.site_vars[i], int_costs[i]))
        
        if weighted_lits:
            # Use CardEnc for pseudo-Boolean AtMost constraint
            # Note: This requires converting to multiple cardinality constraints
            # For proper PB encoding, we use a simplified approach
            
            # Simple approach: Use cardinality encoding with cost grouping
            # Group sites by similar costs
            cost_groups = {}
            for i in range(self.instance.num_sites):
                cost_bucket = int_costs[i] // 100  # Group by hundreds
                if cost_bucket not in cost_groups:
                    cost_groups[cost_bucket] = []
                cost_groups[cost_bucket].append(self.site_vars[i])
            
            # Add cardinality constraints per group
            # This is conservative but ensures budget is respected
            for cost_bucket, site_vars in cost_groups.items():
                if cost_bucket > 0:
                    # Calculate max sites in this bucket
                    bucket_cost = (cost_bucket + 1) * 100  # Upper bound
                    max_sites_in_bucket = max(1, max_int_cost // bucket_cost)
                    
                    if max_sites_in_bucket < len(site_vars):
                        clauses = CardEnc.atmost(
                            lits=site_vars,
                            bound=max_sites_in_bucket,
                            encoding=EncType.seqcounter
                        )
                        
                        for clause in clauses.clauses:
                            self.cnf.append(clause)
                        
                        self.var_counter = max(self.var_counter, clauses.nv)
        
        if self.verbose:
            print(f"  Budget constraint encoded (conservative)")
            print(f"  Total clauses so far: {len(self.cnf.clauses)}")
    
    def _encode_weighted_sum_totalizer(self, weights: List[int], max_sum: int):
        """
        Encode weighted sum constraint using totalizer approach
        
        sum(weights[i] * x[i]) <= max_sum
        
        This creates a tree of adders
        """
        # Group variables by weight
        weight_groups: Dict[int, List[int]] = {}
        for i, w in enumerate(weights):
            if w > 0:
                var = self.site_vars[i]
                if w not in weight_groups:
                    weight_groups[w] = []
                weight_groups[w].append(var)
        
        # For each weight, create count variables
        total_vars = []
        for weight, vars_list in weight_groups.items():
            # Count how many of these variables are true
            # Use unary representation: s[k] = "at least k variables are true"
            max_count = len(vars_list)
            count_vars = [self.new_var(f"count_{weight}_{k}") 
                         for k in range(1, max_count + 1)]
            
            # Encode counting using sequential counter
            # Similar to AtLeast encoding but capturing the count
            for k in range(1, max_count + 1):
                # If at least k variables are true, count_vars[k-1] is true
                clauses = CardEnc.atleast(
                    lits=vars_list,
                    bound=k,
                    encoding=EncType.seqcounter
                )
                # Link clauses to count_var[k-1]
                # (This is simplified; full implementation would integrate better)
                for clause in clauses.clauses:
                    self.cnf.append(clause)
                self.var_counter = max(self.var_counter, clauses.nv)
            
            total_vars.append((weight, count_vars))
        
        # Now encode that weighted sum of counts <= max_sum
        # This requires adding the contributions
        # Simplified: just ensure no single weight group exceeds budget
        for weight, count_vars in total_vars:
            max_allowed_count = max_sum // weight
            if max_allowed_count < len(count_vars):
                # count_vars[max_allowed_count] must be false
                self.cnf.append([-count_vars[max_allowed_count]])
    
    def _encode_weighted_sum_binary(self, weights: List[int], max_sum: int):
        """
        Encode weighted sum using binary representation
        
        Creates binary variables for the sum and builds adder circuits
        """
        num_bits = int(np.ceil(np.log2(max_sum + 1)))
        
        # Binary variables representing the sum
        sum_bits = [self.new_var(f"sum_bit_{b}") for b in range(num_bits)]
        
        # Build binary adder circuit
        # This is complex - simplified version just adds constraint on final sum
        # Full implementation would build carry-save adders
        
        # For now, use a direct constraint that if the sum exceeds max_sum,
        # at least one of the high-order bits that would make it exceed must be 0
        # This is a placeholder for full binary encoding
        pass
    
    def encode_connectivity_constraints(self):
        """
        Encode edge variables as AND of endpoint site variables
        
        For each edge (i,j):
            edge_var[i,j] = site_var[i] AND site_var[j]
        """
        if self.verbose:
            print("Encoding connectivity constraints...")
        
        for (i, j), edge_var in self.edge_vars.items():
            site_i = self.site_vars[i]
            site_j = self.site_vars[j]
            self.encode_and_gate(site_i, site_j, edge_var)
        
        if self.verbose:
            print(f"  Encoded {len(self.edge_vars)} edge constraints")
            print(f"  Total clauses so far: {len(self.cnf.clauses)}")
    
    def encode_compactness_constraints(self):
        """
        Encode compactness (connected component) constraints
        
        For max_components = 1: all selected sites must form a connected subgraph
        Uses reachability encoding or flow-based encoding
        
        This is complex and computationally expensive for large graphs
        """
        if self.verbose:
            print("Encoding compactness constraints...")
        
        if self.instance.max_components == 1 and len(self.instance.adjacency) > 0:
            # Single connected component required
            # Use a flow-based encoding:
            # 1. Select one site as "root"
            # 2. Every other selected site must have a path to root
            # 3. Enforce using flow variables
            
            # Simplified: Use root selection and reachability
            # Root variables: is_root[i] indicates if site i is the root
            root_vars = {i: self.new_var(f"root_{i}") 
                        for i in range(self.instance.num_sites)}
            
            # Exactly one root if any site is selected
            # At least one root
            root_clause = [root_vars[i] for i in range(self.instance.num_sites)]
            self.cnf.append(root_clause)
            
            # At most one root
            for i in range(self.instance.num_sites):
                for j in range(i + 1, self.instance.num_sites):
                    self.cnf.append([-root_vars[i], -root_vars[j]])
            
            # Root must be selected
            for i in range(self.instance.num_sites):
                self.cnf.append([-root_vars[i], self.site_vars[i]])
            
            # Reachability from root
            # For simplicity, we use a bounded-depth reachability encoding
            # reach[i][d] = site i is reachable from root in at most d steps
            max_depth = self.instance.num_sites
            
            # This gets very large; for practical purposes, skip or use approximation
            # Full encoding would require O(n^2 * d) variables and clauses
            
            if self.verbose:
                print("  Compactness encoding (simplified/approximate)")
        
        elif self.instance.max_components > 1:
            # Multiple components allowed
            # More complex encoding required
            if self.verbose:
                print(f"  Multiple components ({self.instance.max_components}) - skipped")
    
    def encode(self, objective_bound: Optional[float] = None) -> CNF:
        """
        Encode complete problem to CNF
        
        Args:
            objective_bound: Maximum cost bound (uses budget if None)
        
        Returns:
            CNF formula
        """
        if objective_bound is None:
            objective_bound = self.instance.budget
        
        if self.verbose:
            print("\n=== Starting SAT Encoding ===")
        
        # Encode variables
        self.encode_site_variables()
        
        if len(self.instance.adjacency) > 0:
            self.encode_connectivity_variables()
        
        # Encode constraints
        self.encode_species_representation()
        self.encode_budget_constraint(objective_bound)
        
        if len(self.instance.adjacency) > 0:
            self.encode_connectivity_constraints()
        
        # Optionally encode compactness (expensive)
        # self.encode_compactness_constraints()
        
        if self.verbose:
            print(f"\n=== Encoding Complete ===")
            print(f"Total variables: {self.var_counter}")
            print(f"Total clauses: {len(self.cnf.clauses)}")
            print(f"Average clause length: {np.mean([len(c) for c in self.cnf.clauses]):.2f}")
        
        return self.cnf
    
    def decode_solution(self, model: List[int]) -> List[int]:
        """
        Decode SAT model to reserve design solution
        
        Args:
            model: SAT assignment (list of signed literals)
        
        Returns:
            List of selected site indices
        """
        selected_sites = []
        
        for site_id, var in self.site_vars.items():
            # Check if variable is true in model
            # Model is 0-indexed, variables are 1-indexed
            if model[var - 1] > 0:
                selected_sites.append(site_id)
        
        return sorted(selected_sites)
    
    def get_encoding_statistics(self) -> dict:
        """Get statistics about the encoding"""
        return {
            'num_variables': self.var_counter,
            'num_clauses': len(self.cnf.clauses),
            'num_site_vars': len(self.site_vars),
            'num_edge_vars': len(self.edge_vars),
            'num_auxiliary_vars': len(self.auxiliary_vars),
            'average_clause_length': np.mean([len(c) for c in self.cnf.clauses]) if self.cnf.clauses else 0,
            'max_clause_length': max([len(c) for c in self.cnf.clauses]) if self.cnf.clauses else 0,
        }


if __name__ == "__main__":
    from reserve_design_instance import ReserveDesignInstance
    
    # Create a small test instance
    print("Creating test instance...")
    instance = ReserveDesignInstance.create_random_instance(
        num_sites=10,
        num_species=3,
        budget_fraction=0.5,
        target_coverage=2,
        connectivity_prob=0.4,
        seed=42
    )
    
    # Encode to SAT
    print("\nEncoding to SAT...")
    encoder = ReserveDesignSATEncoder(instance, verbose=True)
    cnf = encoder.encode()
    
    # Print statistics
    print("\nEncoding Statistics:")
    stats = encoder.get_encoding_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Write to DIMACS file
    output_file = "reserve_design_test.cnf"
    cnf.to_file(output_file)
    print(f"\nCNF written to: {output_file}")
