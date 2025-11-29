#!/usr/bin/env python3
"""Quick check of BQM variable naming patterns"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comprehensive_embedding_and_solving_benchmark import build_patch_cqm, cqm_to_bqm_wrapper

# Build small problem
cqm, _ = build_patch_cqm(5)
bqm, _ = cqm_to_bqm_wrapper(cqm, "test")

# Analyze variable names
y_vars = [v for v in bqm.variables if v.startswith("Y_")]
slack_vars = [v for v in bqm.variables if "slack" in v.lower() or not v.startswith("Y_")]
other_vars = [v for v in bqm.variables if not v.startswith("Y_")]

print(f"Total variables: {len(bqm.variables)}")
print(f"Y variables: {len(y_vars)}")
print(f"Other variables: {len(other_vars)}")
print(f"\nSample Y variables: {y_vars[:5]}")
print(f"Sample other variables: {other_vars[:10]}")
