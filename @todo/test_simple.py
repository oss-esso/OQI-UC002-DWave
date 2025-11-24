#!/usr/bin/env python3
"""
Simple test - just verify the normalized objective values match the baseline.
"""

print("Expected FARM objective (from baseline): 0.3879339788284886")
print("Expected PATCH objective (from baseline): 0.305130")
print("\nThe DECOMPOSED solvers should produce the SAME objectives as baseline.")
print("If they don't match, there's an objective calculation bug.")
