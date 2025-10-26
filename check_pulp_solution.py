# Quick script to check PuLP solution
import json

with open('Benchmarks/COMPREHENSIVE/Patch_PuLP/config_5_run_1.json') as f:
    data = json.load(f)
    print("PuLP objective:", data['objective_value'])
    print("Note: PuLP doesn't store solution details in this file")
    print("\nWe need to extract from PuLP model results...")
