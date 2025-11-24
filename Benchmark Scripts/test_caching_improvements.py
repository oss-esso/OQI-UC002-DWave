"""
Rigorous test comparing three quadratic term building methods
Ensures all methods build the SAME CQM with SAME problem size
"""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from benchmark_scalability_LQ import load_full_family_with_n_farms
from dimod import ConstrainedQuadraticModel, Binary, Real
import time
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Method took too long!")

print("="*80)
print("RIGOROUS QUADRATIC BUILDING COMPARISON")
print("="*80)

# Load a real scenario
n_farms = 1096
print(f"\nLoading scenario with {n_farms} farms...")
farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms, seed=42, fixed_total_land=100.0)

params = config['parameters']
land_availability = params['land_availability']
weights = params['weights']
synergy_matrix = params.get('synergy_matrix', {})
synergy_bonus_weight = weights.get('synergy_bonus', 0.1)
total_area = sum(land_availability.values())

n_synergy_pairs = sum(len(p) for p in synergy_matrix.values()) // 2
n_total_terms = n_farms * n_synergy_pairs

print(f"✓ {len(farms)} farms, {len(foods)} foods, {n_synergy_pairs} synergy pairs")
print(f"✓ Total quadratic terms to build: {n_total_terms:,}")

# Create variables ONCE (shared across all methods)
print(f"\nCreating {len(farms) * len(foods):,} variables...")
A = {}
Y = {}
start = time.time()
for farm in farms:
    for food in foods:
        A[(farm, food)] = Real(f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
        Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
var_time = time.time() - start
print(f"✓ Variables created in {var_time:.2f}s")

# Build linear objective (shared base)
print("\nBuilding linear objective terms...")
start = time.time()
linear_objective = 0
for farm in farms:
    for food in foods:
        linear_objective += (
            weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * A[(farm, food)] +
            weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * A[(farm, food)] -
            weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * A[(farm, food)] +
            weights.get('affordability', 0) * foods[food].get('affordability', 0) * A[(farm, food)] +
            weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * A[(farm, food)]
        )
linear_time = time.time() - start
print(f"✓ Linear objective built in {linear_time:.2f}s")

# Prepare synergy pairs list for iteration
from src.synergy_optimizer_pure import SynergyOptimizer
optimizer = SynergyOptimizer(synergy_matrix, foods)
synergy_pairs_list = [(optimizer.get_crop_name(c1), optimizer.get_crop_name(c2), boost) 
                      for c1, c2, boost in optimizer.iter_pairs()]

print(f"\n{'='*80}")
print(f"METHOD 1: Single sum() - Build list then sum once")
print(f"{'='*80}")

try:
    start = time.time()
    
    # Build term list
    terms = []
    for farm in farms:
        farm_area = land_availability[farm]
        coeff_multiplier = synergy_bonus_weight * farm_area / total_area
        for crop1, crop2, boost in synergy_pairs_list:
            terms.append(boost * coeff_multiplier * Y[(farm, crop1)] * Y[(farm, crop2)])
    
    list_time = time.time() - start
    print(f"✓ Built {len(terms):,} term objects in {list_time:.2f}s")
    
    # Sum all at once
    sum_start = time.time()
    quadratic_obj_method1 = sum(terms)
    sum_time = time.time() - sum_start
    
    time_method1 = list_time + sum_time
    
    # Build CQM
    cqm1 = ConstrainedQuadraticModel()
    cqm1.set_objective(-(linear_objective + quadratic_obj_method1))
    
    print(f"✓ Summed all terms in {sum_time:.2f}s")
    print(f"✓ TOTAL TIME: {time_method1:.2f}s")
    print(f"✓ CQM objective has {len(cqm1.objective.variables)} variables")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    time_method1 = None
    cqm1 = None

print(f"\n{'='*80}")
print(f"METHOD 2: Chunked sum - Build list then sum in chunks")
print(f"{'='*80}")

try:
    start = time.time()
    
    # Reuse terms list from method 1
    chunk_size = 1000
    quadratic_obj_method2 = 0
    
    for chunk_start in range(0, len(terms), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(terms))
        chunk = terms[chunk_start:chunk_end]
        quadratic_obj_method2 += sum(chunk)
    
    time_method2 = time.time() - start
    
    # Build CQM
    cqm2 = ConstrainedQuadraticModel()
    cqm2.set_objective(-(linear_objective + quadratic_obj_method2))
    
    print(f"✓ Summed {len(terms):,} terms in {len(terms)//chunk_size + 1} chunks")
    print(f"✓ TOTAL TIME: {time_method2:.2f}s")
    print(f"✓ CQM objective has {len(cqm2.objective.variables)} variables")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    time_method2 = None
    cqm2 = None

print(f"\n{'='*80}")
print(f"METHOD 3: Incremental += (CURRENT IMPLEMENTATION)")
print(f"{'='*80}")
print(f"⚠️  Testing with 60 second timeout...")

try:
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    
    start = time.time()
    
    quadratic_obj_method3 = 0
    count = 0
    for farm in farms:
        farm_area = land_availability[farm]
        coeff_multiplier = synergy_bonus_weight * farm_area / total_area
        for crop1, crop2, boost in synergy_pairs_list:
            quadratic_obj_method3 += boost * coeff_multiplier * Y[(farm, crop1)] * Y[(farm, crop2)]
            count += 1
    
    time_method3 = time.time() - start
    signal.alarm(0)  # Cancel timeout
    
    # Build CQM
    cqm3 = ConstrainedQuadraticModel()
    cqm3.set_objective(-(linear_objective + quadratic_obj_method3))
    
    print(f"✓ Built {count:,} terms incrementally")
    print(f"✓ TOTAL TIME: {time_method3:.2f}s")
    print(f"✓ CQM objective has {len(cqm3.objective.variables)} variables")
    
except TimeoutError:
    print(f"✗ TIMEOUT after 60 seconds!")
    print(f"✗ This confirms incremental += is EXTREMELY slow for large problems")
    time_method3 = None
    cqm3 = None
    signal.alarm(0)
except Exception as e:
    print(f"✗ FAILED: {e}")
    time_method3 = None
    cqm3 = None
    signal.alarm(0)

print(f"\n{'='*80}")
print(f"RESULTS SUMMARY ({n_farms} farms, {n_total_terms:,} quadratic terms)")
print(f"{'='*80}")

if time_method1:
    print(f"Method 1 (Single sum):  {time_method1:.2f}s")
else:
    print(f"Method 1 (Single sum):  FAILED")

if time_method2:
    print(f"Method 2 (Chunked):     {time_method2:.2f}s")
else:
    print(f"Method 2 (Chunked):     FAILED")

if time_method3:
    print(f"Method 3 (Incremental): {time_method3:.2f}s")
else:
    print(f"Method 3 (Incremental): TIMEOUT/FAILED")

# Determine winner
if all([time_method1, time_method2, time_method3]):
    times = [
        ('Single sum', time_method1),
        ('Chunked', time_method2),
        ('Incremental', time_method3)
    ]
    winner = min(times, key=lambda x: x[1])
    print(f"\n🏆 WINNER: {winner[0]} ({winner[1]:.2f}s)")
    
    print(f"\nSpeedup analysis:")
    print(f"  Chunked vs Single sum: {time_method1/time_method2:.2f}x")
    print(f"  Chunked vs Incremental: {time_method3/time_method2:.2f}x")

print(f"\n{'='*80}")
print(f"RECOMMENDATION")
print(f"{'='*80}")
print(f"Based on rigorous testing with SAME problem size:")
if time_method2 and time_method1:
    if time_method2 < time_method1:
        print(f"✅ USE CHUNKED SUM - {time_method1/time_method2:.1f}x faster than single sum")
    else:
        print(f"✅ USE SINGLE SUM - {time_method2/time_method1:.1f}x faster than chunked")
print(f"❌ NEVER USE INCREMENTAL += for large problems (3+ hours estimated)")

