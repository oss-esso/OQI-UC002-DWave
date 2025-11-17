#!/usr/bin/env python3
"""
3-Period Crop Rotation Benchmark Script

This script benchmarks the 3-period crop rotation optimization problem
with different numbers of plots, testing multiple solvers:

ROTATION Scenario (3-period binary):
- Gurobi (PuLP): BIP solver for 3-period binary plot assignments  
- D-Wave CQM: Quantum-classical hybrid for constrained quadratic models
- Gurobi QUBO: Native QUBO solver after CQM→BQM conversion
- D-Wave BQM: Quantum annealer with higher QPU utilization

Total solver configurations tested: 4

The implementation follows the binary formulation from crop_rotation.tex:
- Time periods: t ∈ {1, 2, 3}
- Variables: Y_{p,c,t} for plot p, crop c, period t
- Objective: Linear crop values + quadratic rotation synergy
- Constraints: Per-period plot assignments and food group constraints
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import patch generator (plots/patches for rotation model)
from Utils.patch_sampler import generate_farms as generate_patches
from src.scenarios import load_food_data

# Import rotation solver
import solver_runner_ROTATION as solver_runner

from dimod import cqm_to_bqm

# Benchmark configurations - number of plots to test
BENCHMARK_CONFIGS = [5, 10, 15]

# Number of runs per configuration
NUM_RUNS = 1

# Default gamma (rotation synergy weight)
DEFAULT_GAMMA = 0.1

