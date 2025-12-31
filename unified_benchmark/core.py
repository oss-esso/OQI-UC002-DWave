"""
Core utilities for the unified benchmark.

Contains:
- JSON schema definition
- Run entry creation and validation
- Benchmark logging utilities
- Common constants and types
"""

import os
import sys
import json
import socket
import subprocess
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field

# Schema version for JSON output
SCHEMA_VERSION = "1.0"

# Benchmark modes
MODES = [
    "gurobi-true-ground-truth",
    "qpu-native-6-family",
    "qpu-hierarchical-aggregated",
    "qpu-hybrid-27-food",
]

# Default timeout
DEFAULT_TIMEOUT = 600  # seconds

# Default MIQP parameters (from formulations.tex)
MIQP_PARAMS = {
    "rotation_gamma": 0.2,       # γ_rot: temporal synergy weight
    "spatial_gamma": 0.1,        # γ_spat: spatial synergy weight
    "one_hot_penalty": 3.0,      # λ_oh: soft one-hot penalty
    "diversity_bonus": 0.15,     # λ_div: diversity bonus
    "k_neighbors": 4,            # k: spatial neighbors per farm
    "frustration_ratio": 0.7,    # ratio of negative synergies
    "negative_strength": -0.8,   # strength of negative synergies
    "n_periods": 3,              # T: number of rotation periods
}


def get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_hostname() -> str:
    """Get hostname."""
    return socket.gethostname()


@dataclass
class TimingInfo:
    """Timing breakdown for a benchmark run."""
    total_wall_time: float = 0.0
    model_build_time: float = 0.0
    solve_time: float = 0.0  # Wall time for solve phase
    postprocess_time: float = 0.0
    qpu_access_time: Optional[float] = None  # Total QPU access time (includes overhead)
    qpu_sampling_time: Optional[float] = None  # Pure QPU sampling time
    embedding_time: Optional[float] = None
    refinement_time: Optional[float] = None
    area_normalization_time: float = 0.0
    miqp_recompute_time: float = 0.0
    
    # For speedup calculations (filled in by benchmark runner)
    gurobi_reference_time: Optional[float] = None  # Reference Gurobi solve time
    speedup_vs_wall: Optional[float] = None  # Gurobi time / solve_time (wall)
    speedup_vs_qpu: Optional[float] = None  # Gurobi time / qpu_access_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, filtering None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DecompositionInfo:
    """Decomposition metadata."""
    method: Optional[str] = None
    cluster_sizes: Optional[List[int]] = None
    n_clusters: Optional[int] = None
    iterations: Optional[int] = None
    boundary_sync_stats: Optional[Dict[str, Any]] = None
    farms_per_cluster: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, filtering None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass 
class ConstraintViolations:
    """Constraint violation details."""
    one_hot_violations: int = 0
    rotation_violations: int = 0
    total_violations: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "one_hot_violations": self.one_hot_violations,
            "rotation_violations": self.rotation_violations,
            "total_violations": self.total_violations,
            "details": self.details[:10],  # Limit to first 10
        }


@dataclass
class RunEntry:
    """A single benchmark run entry."""
    # Required fields
    mode: str
    scenario_name: str
    n_farms: int
    n_foods: int
    n_vars: int
    n_periods: int = 3
    
    # Sampler info
    sampler: str = "sa"  # 'qpu' or 'sa'
    backend: str = "SimulatedAnnealingSampler"
    num_reads: Optional[int] = None
    
    # Timing and status
    timeout_s: float = DEFAULT_TIMEOUT
    status: str = "unknown"  # 'optimal', 'feasible', 'timeout', 'error'
    mip_gap: Optional[float] = None
    
    # Objectives
    objective_miqp: Optional[float] = None  # Recomputed true MIQP
    objective_model: Optional[float] = None  # Raw model objective
    
    # Feasibility
    constraint_violations: Optional[ConstraintViolations] = None
    feasible: bool = False
    
    # Timing breakdown
    timing: Optional[TimingInfo] = None
    
    # Decomposition
    decomposition: Optional[DecompositionInfo] = None
    
    # Solution (optional - can be large)
    solution: Optional[Dict[str, int]] = None
    
    # Metadata
    seed: Optional[int] = None
    timestamp_utc: str = ""
    git_commit: Optional[str] = None
    hostname: str = ""
    area_constant: float = 1.0
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()
        if not self.hostname:
            self.hostname = get_hostname()
        if not self.git_commit:
            self.git_commit = get_git_commit()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "mode": self.mode,
            "scenario_name": self.scenario_name,
            "n_farms": self.n_farms,
            "n_foods": self.n_foods,
            "n_vars": self.n_vars,
            "n_periods": self.n_periods,
            "sampler": self.sampler,
            "backend": self.backend,
            "timeout_s": self.timeout_s,
            "status": self.status,
            "objective_miqp": self.objective_miqp,
            "objective_model": self.objective_model,
            "feasible": self.feasible,
            "timestamp_utc": self.timestamp_utc,
            "hostname": self.hostname,
            "area_constant": self.area_constant,
        }
        
        # Add optional fields
        if self.num_reads is not None:
            d["num_reads"] = self.num_reads
        if self.mip_gap is not None:
            d["mip_gap"] = self.mip_gap
        if self.seed is not None:
            d["seed"] = self.seed
        if self.git_commit is not None:
            d["git_commit"] = self.git_commit
        if self.error_message is not None:
            d["error_message"] = self.error_message
        if self.timing is not None:
            d["timing"] = self.timing.to_dict()
        if self.decomposition is not None:
            d["decomposition"] = self.decomposition.to_dict()
        if self.constraint_violations is not None:
            d["constraint_violations"] = self.constraint_violations.to_dict()
        
        # Don't include full solution by default (too large)
        # Can be enabled via separate flag if needed
        
        return d


def create_run_entry(
    mode: str,
    scenario_name: str,
    n_farms: int,
    n_foods: int,
    n_periods: int = 3,
    **kwargs
) -> RunEntry:
    """
    Create a new run entry with defaults.
    
    Args:
        mode: Benchmark mode (one of MODES)
        scenario_name: Name of the scenario
        n_farms: Number of farms
        n_foods: Number of foods
        n_periods: Number of time periods
        **kwargs: Additional fields to set
    
    Returns:
        RunEntry instance
    """
    n_vars = n_farms * n_foods * n_periods
    entry = RunEntry(
        mode=mode,
        scenario_name=scenario_name,
        n_farms=n_farms,
        n_foods=n_foods,
        n_vars=n_vars,
        n_periods=n_periods,
    )
    
    # Set additional kwargs
    for key, value in kwargs.items():
        if hasattr(entry, key):
            setattr(entry, key, value)
    
    return entry


def validate_run_entry(entry: Dict[str, Any]) -> List[str]:
    """
    Validate a run entry dict.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    required = ["mode", "scenario_name", "n_farms", "n_foods", "n_vars", "status"]
    for field in required:
        if field not in entry:
            errors.append(f"Missing required field: {field}")
    
    if "mode" in entry and entry["mode"] not in MODES:
        errors.append(f"Invalid mode: {entry['mode']}. Must be one of {MODES}")
    
    if "sampler" in entry and entry["sampler"] not in ["qpu", "sa"]:
        errors.append(f"Invalid sampler: {entry['sampler']}. Must be 'qpu' or 'sa'")
    
    if "status" in entry and entry["status"] not in ["optimal", "feasible", "timeout", "error"]:
        errors.append(f"Invalid status: {entry['status']}")
    
    return errors


class BenchmarkLogger:
    """
    Logging utility for benchmark runs.
    
    Provides structured logging with consistent formatting.
    """
    
    def __init__(self, name: str = "unified_benchmark", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def scenario_load(self, name: str, n_farms: int, n_foods: int):
        """Log scenario loading."""
        self.logger.info(f"Loaded scenario '{name}': {n_farms} farms × {n_foods} foods")
    
    def area_normalization(self, constant: float, n_farms: int):
        """Log area normalization."""
        self.logger.info(f"Equal areas: {n_farms} farms × {constant:.2f} = {n_farms * constant:.2f} total")
    
    def model_build_start(self, mode: str, n_vars: int):
        """Log model build start."""
        self.logger.info(f"Building model ({mode}): {n_vars} variables")
    
    def model_build_done(self, time_s: float):
        """Log model build completion."""
        self.logger.info(f"Model built in {time_s:.2f}s")
    
    def solve_start(self, sampler: str, timeout: float):
        """Log solve start."""
        self.logger.info(f"Solving with {sampler} (timeout: {timeout}s)")
    
    def solve_done(self, status: str, time_s: float, objective: Optional[float] = None):
        """Log solve completion."""
        obj_str = f", objective={objective:.4f}" if objective is not None else ""
        self.logger.info(f"Solve complete: status={status}, time={time_s:.2f}s{obj_str}")
    
    def refinement_start(self, method: str):
        """Log refinement start."""
        self.logger.info(f"Starting refinement ({method})")
    
    def refinement_done(self, time_s: float):
        """Log refinement completion."""
        self.logger.info(f"Refinement complete in {time_s:.2f}s")
    
    def miqp_recompute(self, objective: float, time_s: float):
        """Log MIQP objective recomputation."""
        self.logger.info(f"MIQP objective recomputed: {objective:.4f} ({time_s:.3f}s)")
    
    def constraint_check(self, violations: int, feasible: bool):
        """Log constraint check."""
        status = "FEASIBLE" if feasible else f"INFEASIBLE ({violations} violations)"
        self.logger.info(f"Constraint check: {status}")
    
    def json_write(self, path: str, n_runs: int):
        """Log JSON write."""
        self.logger.info(f"Writing {n_runs} run(s) to {path}")
    
    def error(self, message: str):
        """Log error."""
        self.logger.error(message)
    
    def warning(self, message: str):
        """Log warning."""
        self.logger.warning(message)
    
    def info(self, message: str):
        """Log info."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug."""
        self.logger.debug(message)


def save_benchmark_results(
    runs: List[RunEntry],
    output_path: str,
    include_solutions: bool = False
) -> None:
    """
    Save benchmark results to JSON file.
    
    Args:
        runs: List of RunEntry objects
        output_path: Path to output JSON file
        include_solutions: Whether to include full solutions (can be large)
    """
    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [r.to_dict() for r in runs],
    }
    
    # Optionally add solutions
    if include_solutions:
        for i, run in enumerate(runs):
            if run.solution:
                output["runs"][i]["solution"] = run.solution
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def load_benchmark_results(input_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)
