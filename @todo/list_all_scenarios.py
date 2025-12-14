#!/usr/bin/env python3
"""
Complete Scenarios Inventory - Maps all available scenarios from scenarios.py
to hardness analysis findings and QPU performance expectations.
"""

import json
from pathlib import Path

# All scenarios from src/scenarios.py
ALL_SCENARIOS = {
    # =========================================================================
    # SYNTHETIC SMALL-SCALE SCENARIOS (6-160 variables)
    # For QPU embedding testing - mostly tractable for Gurobi
    # =========================================================================
    'micro_6': {
        'description': 'Micro scenario - 6 variables total',
        'n_vars': 6,
        'estimated_farms': 1,
        'n_foods': 2,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'FAST (< 1s)',
        'qpu_expected': 'Direct QPU embedding possible',
        'hardness': 'trivial',
    },
    'micro_12': {
        'description': 'Micro scenario - 12 variables',
        'n_vars': 12,
        'estimated_farms': 2,
        'n_foods': 2,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'FAST (< 1s)',
        'qpu_expected': 'Direct QPU embedding possible',
        'hardness': 'trivial',
    },
    'tiny_24': {
        'description': 'Tiny scenario - 24 variables',
        'n_vars': 24,
        'estimated_farms': 4,
        'n_foods': 2,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'FAST (< 1s)',
        'qpu_expected': 'Direct QPU embedding possible',
        'hardness': 'easy',
    },
    'tiny_40': {
        'description': 'Tiny scenario - 40 variables',
        'n_vars': 40,
        'estimated_farms': 4,
        'n_foods': 3,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'FAST (< 5s)',
        'qpu_expected': 'Direct QPU embedding possible',
        'hardness': 'easy',
    },
    'small_60': {
        'description': 'Small scenario - 60 variables',
        'n_vars': 60,
        'estimated_farms': 5,
        'n_foods': 4,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'FAST (< 10s)',
        'qpu_expected': 'Direct QPU or minor decomposition',
        'hardness': 'easy',
    },
    'small_80': {
        'description': 'Small scenario - 80 variables',
        'n_vars': 80,
        'estimated_farms': 5,
        'n_foods': 5,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'MEDIUM (10-60s)',
        'qpu_expected': 'Decomposition recommended',
        'hardness': 'moderate',
    },
    'small_100': {
        'description': 'Small scenario - 100 variables',
        'n_vars': 100,
        'estimated_farms': 6,
        'n_foods': 5,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'MEDIUM (10-100s)',
        'qpu_expected': 'Decomposition required',
        'hardness': 'moderate',
    },
    'medium_120': {
        'description': 'Medium scenario - 120 variables',
        'n_vars': 120,
        'estimated_farms': 7,
        'n_foods': 6,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'MEDIUM-TIMEOUT (60-300s)',
        'qpu_expected': 'Decomposition required',
        'hardness': 'hard',
    },
    'medium_160': {
        'description': 'Medium scenario - 160 variables',
        'n_vars': 160,
        'estimated_farms': 9,
        'n_foods': 6,
        'n_periods': 3,
        'category': 'synthetic_small',
        'gurobi_expected': 'TIMEOUT likely',
        'qpu_expected': 'Decomposition required',
        'hardness': 'hard',
    },
    
    # =========================================================================
    # ROTATION SCENARIOS (quantum-friendly, with rotation constraints)
    # These are the HARD scenarios identified in hardness analysis
    # =========================================================================
    'rotation_micro_25': {
        'description': '5 farms × 6 foods × 3 periods - rotation constraints',
        'n_vars': 90,
        'estimated_farms': 5,
        'n_foods': 6,
        'n_periods': 3,
        'category': 'rotation',
        'gurobi_expected': 'TIMEOUT (> 100s) - confirmed in analysis',
        'qpu_expected': 'clique_decomp: ~18s, 8% gap',
        'hardness': 'HARD - rotation constraints make it NP-hard',
        'analysis_result': {
            'gurobi_time': 210,
            'gurobi_obj': 4.08,
            'qpu_time': 18,
            'qpu_obj': 3.75,
            'speedup': 11.5,
            'gap_pct': 8.2,
        }
    },
    'rotation_small_50': {
        'description': '~10 farms × 6 foods × 3 periods - rotation constraints',
        'n_vars': 180,
        'estimated_farms': 10,
        'n_foods': 6,
        'n_periods': 3,
        'category': 'rotation',
        'gurobi_expected': 'TIMEOUT (> 100s) - at the "10-farm cliff"',
        'qpu_expected': 'spatial_temporal: ~39s, 9.6% gap',
        'hardness': 'VERY HARD - critical threshold',
        'analysis_result': {
            'gurobi_time': 240,
            'gurobi_obj': 7.17,
            'qpu_time': 39,
            'qpu_obj': 6.49,
            'speedup': 6.2,
            'gap_pct': 9.6,
        }
    },
    'rotation_medium_100': {
        'description': '~20 farms × 6 foods × 3 periods - rotation constraints',
        'n_vars': 360,
        'estimated_farms': 20,
        'n_foods': 6,
        'n_periods': 3,
        'category': 'rotation',
        'gurobi_expected': 'TIMEOUT (300s) - always fails',
        'qpu_expected': 'clique_decomp: ~57s, 13% gap',
        'hardness': 'EXTREMELY HARD - beyond classical',
        'analysis_result': {
            'gurobi_time': 300,
            'gurobi_obj': 14.89,
            'qpu_time': 57,
            'qpu_obj': 12.98,
            'speedup': 5.2,
            'gap_pct': 12.9,
        }
    },
    'rotation_large_200': {
        'description': '~35+ farms × 6 foods × 3 periods - rotation constraints',
        'n_vars': 630,
        'estimated_farms': 35,
        'n_foods': 6,
        'n_periods': 3,
        'category': 'rotation',
        'gurobi_expected': 'TIMEOUT (300s) - intractable',
        'qpu_expected': 'Hierarchical decomposition required',
        'hardness': 'EXTREMELY HARD',
    },
    
    # =========================================================================
    # LARGE-SCALE ROTATION SCENARIOS (for hierarchical quantum solver)
    # Require 27 foods with aggregation to 6 families
    # =========================================================================
    'rotation_250farms_27foods': {
        'description': '250 farms × 27 foods × 3 periods - massive scale',
        'n_vars': 20250,  # 250 × 27 × 3
        'n_vars_aggregated': 4500,  # 250 × 6 × 3
        'estimated_farms': 250,
        'n_foods': 27,
        'n_periods': 3,
        'category': 'large_rotation',
        'gurobi_expected': 'INTRACTABLE - far beyond limits',
        'qpu_expected': 'Hierarchical QPU only option',
        'hardness': 'INTRACTABLE for classical',
    },
    'rotation_350farms_27foods': {
        'description': '350 farms × 27 foods × 3 periods',
        'n_vars': 28350,
        'n_vars_aggregated': 6300,
        'estimated_farms': 350,
        'n_foods': 27,
        'n_periods': 3,
        'category': 'large_rotation',
        'gurobi_expected': 'INTRACTABLE',
        'qpu_expected': 'Hierarchical QPU only option',
        'hardness': 'INTRACTABLE for classical',
    },
    'rotation_500farms_27foods': {
        'description': '500 farms × 27 foods × 3 periods',
        'n_vars': 40500,
        'n_vars_aggregated': 9000,
        'estimated_farms': 500,
        'n_foods': 27,
        'n_periods': 3,
        'category': 'large_rotation',
        'gurobi_expected': 'INTRACTABLE',
        'qpu_expected': 'Hierarchical QPU only option',
        'hardness': 'INTRACTABLE for classical',
    },
    'rotation_1000farms_27foods': {
        'description': '1000 farms × 27 foods × 3 periods - ultimate scale',
        'n_vars': 81000,
        'n_vars_aggregated': 18000,
        'estimated_farms': 1000,
        'n_foods': 27,
        'n_periods': 3,
        'category': 'large_rotation',
        'gurobi_expected': 'INTRACTABLE',
        'qpu_expected': 'Hierarchical QPU required - may hit QPU limits',
        'hardness': 'INTRACTABLE for classical',
    },
    
    # =========================================================================
    # STANDARD FARM SCENARIOS (from Excel data, no rotation)
    # These are assignment problems - generally tractable
    # =========================================================================
    'simple': {
        'description': 'Simple baseline scenario',
        'category': 'standard',
        'gurobi_expected': 'FAST',
        'hardness': 'easy - no rotation',
    },
    'intermediate': {
        'description': 'Intermediate scenario',
        'category': 'standard',
        'gurobi_expected': 'FAST-MEDIUM',
        'hardness': 'easy-moderate',
    },
    '30farms': {
        'description': '30 farms from sampler',
        'estimated_farms': 30,
        'category': 'standard',
        'gurobi_expected': 'MEDIUM (without rotation)',
        'hardness': 'moderate without rotation constraints',
    },
    '60farms': {
        'description': '60 farms - ~600 pairs',
        'estimated_farms': 60,
        'category': 'standard',
        'gurobi_expected': 'MEDIUM-SLOW (without rotation)',
        'hardness': 'moderate-hard without rotation',
    },
    '90farms': {
        'description': '90 farms - ~900 pairs',
        'estimated_farms': 90,
        'category': 'standard',
        'gurobi_expected': 'SLOW (without rotation)',
        'hardness': 'hard without rotation',
    },
    '250farms': {
        'description': '250 farms - ~2500 pairs',
        'estimated_farms': 250,
        'category': 'standard',
        'gurobi_expected': 'TIMEOUT likely (without rotation)',
        'hardness': 'very hard even without rotation',
    },
    '350farms': {
        'description': '350 farms - ~3500 pairs',
        'estimated_farms': 350,
        'category': 'standard',
        'gurobi_expected': 'TIMEOUT (without rotation)',
        'hardness': 'very hard',
    },
    '500farms_full': {
        'description': '500 farms with full 27 foods',
        'estimated_farms': 500,
        'n_foods': 27,
        'category': 'standard_large',
        'gurobi_expected': 'INTRACTABLE',
        'hardness': 'extremely hard',
    },
    '1000farms_full': {
        'description': '1000 farms with full 27 foods',
        'estimated_farms': 1000,
        'n_foods': 27,
        'category': 'standard_large',
        'gurobi_expected': 'INTRACTABLE',
        'hardness': 'extremely hard',
    },
    '2000farms_full': {
        'description': '2000 farms with full 27 foods',
        'estimated_farms': 2000,
        'n_foods': 27,
        'category': 'standard_large',
        'gurobi_expected': 'INTRACTABLE',
        'hardness': 'extremely hard',
    },
}

# Categorize by hardness based on analysis
HARDNESS_CATEGORIES = {
    'trivial': ['micro_6', 'micro_12', 'tiny_24'],
    'easy': ['tiny_40', 'small_60', 'simple'],
    'moderate': ['small_80', 'small_100', 'intermediate', '30farms'],
    'hard': ['medium_120', 'medium_160', '60farms', '90farms'],
    'very_hard_rotation': ['rotation_micro_25', 'rotation_small_50'],
    'extremely_hard_rotation': ['rotation_medium_100', 'rotation_large_200'],
    'intractable': ['rotation_250farms_27foods', 'rotation_350farms_27foods', 
                    'rotation_500farms_27foods', 'rotation_1000farms_27foods',
                    '250farms', '350farms', '500farms_full', '1000farms_full', '2000farms_full'],
}

# Best solver recommendations
SOLVER_RECOMMENDATIONS = {
    'gurobi': ['micro_6', 'micro_12', 'tiny_24', 'tiny_40', 'small_60', 'simple', 'intermediate'],
    'qpu_direct': ['micro_6', 'micro_12', 'tiny_24', 'tiny_40'],
    'clique_decomp': ['rotation_micro_25', 'rotation_small_50', 'small_80', 'small_100'],
    'spatial_temporal': ['rotation_small_50', 'rotation_medium_100', 'medium_120', 'medium_160'],
    'hierarchical_qpu': ['rotation_medium_100', 'rotation_large_200', 'rotation_250farms_27foods',
                         'rotation_350farms_27foods', 'rotation_500farms_27foods', 'rotation_1000farms_27foods'],
}

# Output
OUTPUT_DIR = Path(__file__).parent / 'significant_scenarios'
OUTPUT_DIR.mkdir(exist_ok=True)

# Save complete inventory
with open(OUTPUT_DIR / 'complete_scenarios_inventory.json', 'w') as f:
    json.dump({
        'scenarios': ALL_SCENARIOS,
        'hardness_categories': HARDNESS_CATEGORIES,
        'solver_recommendations': SOLVER_RECOMMENDATIONS,
    }, f, indent=2)

print("="*80)
print("COMPLETE SCENARIOS INVENTORY")
print("="*80)

# Summary by category
print("\n📊 SCENARIOS BY CATEGORY:")
for category in ['synthetic_small', 'rotation', 'large_rotation', 'standard', 'standard_large']:
    scenarios = [k for k, v in ALL_SCENARIOS.items() if v.get('category') == category]
    print(f"\n  {category.upper()}: {len(scenarios)} scenarios")
    for s in scenarios:
        info = ALL_SCENARIOS[s]
        vars_str = f"{info.get('n_vars', '?')} vars" if 'n_vars' in info else ''
        print(f"    • {s}: {vars_str} - {info.get('hardness', 'unknown')}")

# Summary by hardness
print("\n" + "="*80)
print("📈 SCENARIOS BY HARDNESS (based on analysis):")
print("="*80)
for hardness, scenarios in HARDNESS_CATEGORIES.items():
    print(f"\n  {hardness.upper()}: {len(scenarios)} scenarios")
    for s in scenarios:
        if s in ALL_SCENARIOS:
            info = ALL_SCENARIOS[s]
            print(f"    • {s}: {info.get('gurobi_expected', 'N/A')}")

# Solver recommendations
print("\n" + "="*80)
print("🔧 SOLVER RECOMMENDATIONS:")
print("="*80)
for solver, scenarios in SOLVER_RECOMMENDATIONS.items():
    print(f"\n  {solver}: {len(scenarios)} scenarios")
    for s in scenarios[:5]:  # Show first 5
        if s in ALL_SCENARIOS:
            info = ALL_SCENARIOS[s]
            print(f"    • {s}")
    if len(scenarios) > 5:
        print(f"    ... and {len(scenarios)-5} more")

# Key rotation scenarios with QPU results
print("\n" + "="*80)
print("⭐ KEY ROTATION SCENARIOS (from hardness analysis):")
print("="*80)

key_scenarios = ['rotation_micro_25', 'rotation_small_50', 'rotation_medium_100', 'rotation_large_200']
print("\n| Scenario | Vars | Gurobi | QPU | Speedup | Gap |")
print("|----------|------|--------|-----|---------|-----|")
for s in key_scenarios:
    info = ALL_SCENARIOS[s]
    result = info.get('analysis_result', {})
    if result:
        print(f"| {s} | {info['n_vars']} | {result.get('gurobi_time', 'N/A')}s | {result.get('qpu_time', 'N/A')}s | {result.get('speedup', 'N/A')}× | {result.get('gap_pct', 'N/A')}% |")
    else:
        print(f"| {s} | {info['n_vars']} | {info['gurobi_expected']} | {info['qpu_expected']} | - | - |")

print(f"\n✓ Complete inventory saved to: {OUTPUT_DIR / 'complete_scenarios_inventory.json'}")
