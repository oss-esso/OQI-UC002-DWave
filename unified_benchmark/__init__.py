"""
Unified Benchmark Package for Crop Rotation Optimization

This package implements a rigorous benchmark suite comparing quantum and classical
approaches to the crop rotation MIQP problem.

Modules:
- core: JSON schema, logging, utilities
- scenarios: Scenario loading with equal-area enforcement  
- miqp_scorer: True MIQP objective recomputation
- gurobi_solver: Ground truth Gurobi solver
- quantum_solvers: SA/QPU solvers (native, hierarchical, hybrid)
"""

__version__ = "1.0.0"

from .core import (
    SCHEMA_VERSION,
    create_run_entry,
    validate_run_entry,
    BenchmarkLogger,
)

from .scenarios import (
    load_scenario,
    get_available_scenarios,
    DEFAULT_AREA_CONSTANT,
)

from .miqp_scorer import (
    compute_miqp_objective,
    check_constraints,
)

__all__ = [
    "SCHEMA_VERSION",
    "create_run_entry",
    "validate_run_entry",
    "BenchmarkLogger",
    "load_scenario",
    "get_available_scenarios",
    "DEFAULT_AREA_CONSTANT",
    "compute_miqp_objective",
    "check_constraints",
]
