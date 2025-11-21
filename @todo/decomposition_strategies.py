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


class DecompositionStrategy(str, Enum):
    """Available decomposition strategies."""
    CURRENT_HYBRID = "current_hybrid"
    BENDERS = "benders"
    BENDERS_QPU = "benders_qpu"
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
        
        # Convert to format expected by existing solver
        farms_list = list(farms.keys())
        
        result = solve_farm_with_hybrid_decomposition(
            farms_list=farms_list,
            foods=foods,
            food_groups=food_groups,
            land_availability=farms,
            config=config,
            token=kwargs.get('dwave_token'),
            **kwargs
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
        
        return solve_with_benders(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            max_iterations=kwargs.get('max_iterations', 50),
            gap_tolerance=kwargs.get('gap_tolerance', 1e-4),
            time_limit=kwargs.get('time_limit', 300.0)
        )


class BendersQPUStrategy(BaseDecompositionStrategy):
    """Benders decomposition with QPU integration for master problem."""
    
    def __init__(self):
        super().__init__(
            name="Benders Decomposition (QPU)",
            description="Master problem uses QPU/Hybrid, subproblem classical LP"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_benders_qpu import solve_with_benders_qpu
        
        return solve_with_benders_qpu(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=kwargs.get('dwave_token'),
            max_iterations=kwargs.get('max_iterations', 50),
            gap_tolerance=kwargs.get('gap_tolerance', 1e-4),
            time_limit=kwargs.get('time_limit', 300.0),
            use_qpu_for_master=kwargs.get('use_qpu_for_master', True)
        )


class DantzigWolfeStrategy(BaseDecompositionStrategy):
    """Dantzig-Wolfe decomposition strategy (classical only)."""
    
    def __init__(self):
        super().__init__(
            name="Dantzig-Wolfe Decomposition",
            description="Column generation with restricted master and pricing subproblem (classical)"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_dantzig_wolfe import solve_with_dantzig_wolfe
        
        return solve_with_dantzig_wolfe(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            max_iterations=kwargs.get('max_iterations', 50),
            time_limit=kwargs.get('time_limit', 300.0)
        )


class DantzigWolfeQPUStrategy(BaseDecompositionStrategy):
    """Dantzig-Wolfe decomposition with QPU integration for pricing."""
    
    def __init__(self):
        super().__init__(
            name="Dantzig-Wolfe Decomposition (QPU)",
            description="RMP classical, pricing subproblem uses QPU/Hybrid"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_dantzig_wolfe_qpu import solve_with_dantzig_wolfe_qpu
        
        return solve_with_dantzig_wolfe_qpu(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=kwargs.get('dwave_token'),
            max_iterations=kwargs.get('max_iterations', 50),
            time_limit=kwargs.get('time_limit', 300.0),
            use_qpu_for_pricing=kwargs.get('use_qpu_for_pricing', True)
        )


class ADMMStrategy(BaseDecompositionStrategy):
    """ADMM decomposition strategy."""
    
    def __init__(self):
        super().__init__(
            name="ADMM",
            description="Alternating Direction Method of Multipliers with consensus"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_admm import solve_with_admm
        
        return solve_with_admm(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            max_iterations=kwargs.get('max_iterations', 100),
            rho=kwargs.get('rho', 1.0),
            tolerance=kwargs.get('tolerance', 1e-3),
            time_limit=kwargs.get('time_limit', 300.0)
        )


class ADMMQPUStrategy(BaseDecompositionStrategy):
    """ADMM with QPU integration for Y subproblem."""
    
    def __init__(self):
        super().__init__(
            name="ADMM (QPU)",
            description="ADMM with QPU/Hybrid for binary Y subproblem"
        )
    
    def solve(self, farms: Dict, foods: List[str], food_groups: Dict, config: Dict, **kwargs) -> Dict:
        from decomposition_admm_qpu import solve_with_admm_qpu
        
        return solve_with_admm_qpu(
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=kwargs.get('dwave_token'),
            max_iterations=kwargs.get('max_iterations', 50),
            rho=kwargs.get('rho', 1.0),
            tolerance=kwargs.get('tolerance', 1e-4),
            use_qpu_for_y=kwargs.get('use_qpu_for_y', True)
        )


class DecompositionFactory:
    """Factory for creating decomposition strategy instances."""
    
    _strategies = {
        DecompositionStrategy.CURRENT_HYBRID: CurrentHybridStrategy,
        DecompositionStrategy.BENDERS: BendersStrategy,
        DecompositionStrategy.BENDERS_QPU: BendersQPUStrategy,
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
