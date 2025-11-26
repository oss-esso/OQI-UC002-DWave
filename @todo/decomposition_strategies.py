"""
Unified Decomposition Strategy Interface

Provides a factory pattern for all decomposition strategies:
- Current Hybrid (Gurobi Relaxation + QPU)
- Benders Decomposition
- Dantzig-Wolfe Decomposition
- ADMM Decomposition

All strategies implement a common interface for easy benchmarking.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from enum import Enum
from result_format_compat import convert_to_old_format


class DecompositionStrategy(str, Enum):
    """Available decomposition strategies."""
    CURRENT_HYBRID = "current_hybrid"
    BENDERS = "benders"
    BENDERS_QPU = "benders_qpu"
    BENDERS_HIERARCHICAL = "benders_hierarchical"
    DANTZIG_WOLFE = "dantzig_wolfe"
    DANTZIG_WOLFE_QPU = "dantzig_wolfe_qpu"
    ADMM = "admm"
    ADMM_QPU = "admm_qpu"


class BaseDecompositionStrategy(ABC):
    """Abstract base class for decomposition strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def solve(
        self,
        farms: Dict[str, float],
        foods: List[str],
        food_groups: Dict,
        config: Dict,
        **kwargs
    ) -> Dict:
        """
        Solve the farm allocation problem using this strategy.
        
        Args:
            farms: Dictionary of farm names to land availability
            foods: List of food names
            food_groups: Dictionary of food group constraints
            config: Configuration dictionary with parameters
            **kwargs: Strategy-specific parameters
        
        Returns:
            Standardized result dictionary
        """
        pass
    
    def __str__(self):
        return f"{self.name}: {self.description}"


class CurrentHybridStrategy(BaseDecompositionStrategy):
    """Current hybrid strategy: Gurobi relaxation + QPU binary solving."""
    
    def __init__(self):
        super().__init__(
            name="Current Hybrid",
            description="Gurobi solves continuous relaxation, QPU solves binary subproblem"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from solver_runner_DECOMPOSED import solve_farm_with_hybrid_decomposition
        from result_formatter import format_decomposition_result, validate_solution_constraints
        
        # Convert to format expected by existing solver
        farms_list = list(farms.keys())
        
        raw_result = solve_farm_with_hybrid_decomposition(
            farms=farms_list,
            foods=foods,
            food_groups=food_groups,
            config=config,
            token=kwargs.get('dwave_token'),
            **kwargs
        )
        
        # Validate solution
        validation = validate_solution_constraints(
            raw_result['solution'], farms, foods, food_groups, farms, config, 'farm'
        )
        
        # Wrap in standard format using format_decomposition_result
        result = format_decomposition_result(
            strategy_name='current_hybrid',
            scenario_type='farm',
            n_units=len(farms),
            n_foods=len(foods),
            total_area=sum(farms.values()),
            objective_value=raw_result.get('final_objective', raw_result.get('objective_value', 0.0)),
            solution=raw_result['solution'],
            solve_time=raw_result['solve_time'],
            num_iterations=1,
            is_feasible=validation['is_feasible'],
            validation_results=validation,
            num_variables=len(farms) * len(foods) * 2,
            num_constraints=len(farms) + len(food_groups) * 2,
            status='Optimal' if validation['is_feasible'] else 'failed',
            gurobi_time=raw_result.get('gurobi_time', 0),
            qpu_time=raw_result.get('qpu_time', 0),
            relaxation_objective=raw_result.get('relaxation_objective', 0)
        )
        
        return result


class BendersStrategy(BaseDecompositionStrategy):
    """Benders decomposition strategy (classical only)."""
    
    def __init__(self):
        super().__init__(
            name="Benders Decomposition",
            description="Master problem (Y) with subproblem (A) and optimality cuts (classical)"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_benders import solve_with_benders
        
        result = solve_with_benders(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            max_iterations=kwargs.get('max_iterations', 50),
            gap_tolerance=kwargs.get('gap_tolerance', 1e-4),
            time_limit=kwargs.get('time_limit', 300.0)
        )
        return result  # Return new format directly


class BendersQPUStrategy(BaseDecompositionStrategy):
    """Benders decomposition with QPU integration for master problem."""
    
    def __init__(self):
        super().__init__(
            name="Benders Decomposition (QPU)",
            description="Master problem uses QPU/Hybrid, subproblem classical LP"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_benders_qpu import solve_with_benders_qpu
        
        result = solve_with_benders_qpu(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=kwargs.get('dwave_token', None),
            max_iterations=kwargs.get('max_iterations', 50),
            gap_tolerance=kwargs.get('gap_tolerance', 1e-4),
            time_limit=kwargs.get('time_limit', 300.0),
            use_qpu_for_master=kwargs.get('use_qpu_for_master', True),
            no_improvement_cutoff=kwargs.get('no_improvement_cutoff', 3),
            num_reads=kwargs.get('num_reads', 1000),
            annealing_time=kwargs.get('annealing_time', 20)
        )
        return result  # Return new format directly


class BendersHierarchicalStrategy(BaseDecompositionStrategy):
    """Hierarchical Benders decomposition with graph partitioning for large problems."""
    
    def __init__(self):
        super().__init__(
            name="Benders Hierarchical (QPU)",
            description="Hierarchical graph partitioning for large-scale problems, inspired by QAOA-in-QAOA"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_benders_hierarchical import solve_with_benders_hierarchical
        
        result = solve_with_benders_hierarchical(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=kwargs.get('dwave_token', None),
            max_iterations=kwargs.get('max_iterations', 50),
            gap_tolerance=kwargs.get('gap_tolerance', 1e-4),
            time_limit=kwargs.get('time_limit', 600.0),
            max_embeddable_vars=kwargs.get('max_embeddable_vars', 150),
            use_qpu=kwargs.get('use_qpu', True),
            num_reads=kwargs.get('num_reads', 200),
            annealing_time=kwargs.get('annealing_time', 20)
        )
        return result  # Return new format directly


class DantzigWolfeStrategy(BaseDecompositionStrategy):
    """Dantzig-Wolfe decomposition strategy (classical only)."""
    
    def __init__(self):
        super().__init__(
            name="Dantzig-Wolfe Decomposition",
            description="Column generation with restricted master and pricing subproblem (classical)"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_dantzig_wolfe import solve_with_dantzig_wolfe
        
        result = solve_with_dantzig_wolfe(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            max_iterations=kwargs.get('max_iterations', 50),
            time_limit=kwargs.get('time_limit', 300.0)
        )
        return result  # Return new format directly


class DantzigWolfeQPUStrategy(BaseDecompositionStrategy):
    """Dantzig-Wolfe decomposition with QPU integration for pricing."""
    
    def __init__(self):
        super().__init__(
            name="Dantzig-Wolfe Decomposition (QPU)",
            description="RMP classical, pricing subproblem uses QPU/Hybrid"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_dantzig_wolfe_qpu import solve_with_dantzig_wolfe_qpu
        
        result = solve_with_dantzig_wolfe_qpu(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=kwargs.get('dwave_token', None),
            max_iterations=kwargs.get('max_iterations', 50),
            time_limit=kwargs.get('time_limit', 300.0),
            use_qpu_for_pricing=kwargs.get('use_qpu_for_pricing', True)
        )
        return result  # Return new format directly


class ADMMStrategy(BaseDecompositionStrategy):
    """ADMM decomposition strategy."""
    
    def __init__(self):
        super().__init__(
            name="ADMM",
            description="Alternating Direction Method of Multipliers with consensus"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_admm import solve_with_admm
        
        result = solve_with_admm(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            max_iterations=kwargs.get('max_iterations', 10),
            rho=kwargs.get('rho', 10.0),  # Higher penalty for better A-Y consensus
            tolerance=kwargs.get('tolerance', 1e-3),
            time_limit=kwargs.get('time_limit', 300.0)
        )
        return result  # Return new format directly


class ADMMQPUStrategy(BaseDecompositionStrategy):
    """ADMM with QPU integration for Y subproblem."""
    
    def __init__(self):
        super().__init__(
            name="ADMM (QPU)",
            description="ADMM with QPU/Hybrid for binary Y subproblem"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_admm_qpu import solve_with_admm_qpu
        
        result = solve_with_admm_qpu(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=kwargs.get('dwave_token', None),
            max_iterations=kwargs.get('max_iterations', 10),
            rho=kwargs.get('rho', 10.0),  # Higher penalty for better A-Y consensus
            tolerance=kwargs.get('tolerance', 1e-4),
            use_qpu_for_y=kwargs.get('use_qpu_for_y', True)
        )
        return result  # Return new format directly


class DecompositionFactory:
    """Factory for creating decomposition strategy instances."""
    
    _strategies = {
        DecompositionStrategy.CURRENT_HYBRID: CurrentHybridStrategy,
        DecompositionStrategy.BENDERS: BendersStrategy,
        DecompositionStrategy.BENDERS_QPU: BendersQPUStrategy,
        DecompositionStrategy.BENDERS_HIERARCHICAL: BendersHierarchicalStrategy,
        DecompositionStrategy.DANTZIG_WOLFE: DantzigWolfeStrategy,
        DecompositionStrategy.DANTZIG_WOLFE_QPU: DantzigWolfeQPUStrategy,
        DecompositionStrategy.ADMM: ADMMStrategy,
        DecompositionStrategy.ADMM_QPU: ADMMQPUStrategy
    }
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> BaseDecompositionStrategy:
        """
        Get a decomposition strategy by name.
        
        Args:
            strategy_name: Name of the strategy (see DecompositionStrategy enum)
        
        Returns:
            Instance of the requested strategy
        
        Raises:
            ValueError: If strategy name is not recognized
        """
        try:
            strategy_enum = DecompositionStrategy(strategy_name.lower())
            strategy_class = cls._strategies[strategy_enum]
            return strategy_class()
        except (ValueError, KeyError):
            available = ", ".join(s.value for s in DecompositionStrategy)
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available strategies: {available}"
            )
    
    @classmethod
    def get_all_strategies(cls) -> List[BaseDecompositionStrategy]:
        """Get instances of all available strategies."""
        return [strategy_class() for strategy_class in cls._strategies.values()]
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available strategy names."""
        return [s.value for s in DecompositionStrategy]


def solve_with_strategy(
    strategy_name: str,
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    **kwargs
) -> Dict:
    """
    Convenience function to solve using a named strategy.
    
    Args:
        strategy_name: Name of decomposition strategy
        farms: Dictionary of farm names to land availability
        foods: List of food names
        food_groups: Dictionary of food group constraints
        config: Configuration dictionary
        **kwargs: Strategy-specific parameters
    
    Returns:
        Standardized result dictionary
    """
    strategy = DecompositionFactory.get_strategy(strategy_name)
    print(f"\nUsing strategy: {strategy}")
    return strategy.solve(farms, foods, food_groups, config, **kwargs)


# Example usage
if __name__ == "__main__":
    print("Available Decomposition Strategies:")
    print("=" * 60)
    
    for strategy in DecompositionFactory.get_all_strategies():
        print(f"  • {strategy}")
    
    print("\nStrategy names for command-line use:")
    for name in DecompositionFactory.list_strategies():
        print(f"  - {name}")
