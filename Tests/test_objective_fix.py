"""
Quick test script to verify the objective calculation fix.
Tests all 4 benchmark configurations without caching.
"""
from benchmark_scalability_PATCH import run_benchmark
import os

configs = [5, 10, 15, 25]
dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')

print("\n" + "="*120)
print("OBJECTIVE FIX VERIFICATION TEST")
print("="*120)
print(f"{'Config':<10} {'PuLP Obj':<15} {'DWave Obj':<15} {'Diff':<15} {'Gap %':<10} {'Status'}")
print("-"*120)

for n_patches in configs:
    try:
        result = run_benchmark(
            n_patches=n_patches, 
            run_number=1, 
            total_runs=1, 
            dwave_token=dwave_token, 
            cache=None, 
            save_to_cache=False
        )
        
        if result:
            pulp_obj = result.get('pulp_objective', 0)
            dwave_obj = result.get('dwave_objective', 0)
            
            if pulp_obj and dwave_obj:
                diff = abs(pulp_obj - dwave_obj)
                # Calculate gap from best (max for maximization)
                best_obj = max(pulp_obj, dwave_obj)
                pulp_gap = ((best_obj - pulp_obj) / best_obj * 100) if pulp_obj else 0
                dwave_gap = ((best_obj - dwave_obj) / best_obj * 100) if dwave_obj else 0
                
                status = "✅ GOOD" if diff < abs(pulp_obj) * 0.5 else "⚠️ CHECK"
                
                print(f"{n_patches:<10} {pulp_obj:<15.6f} {dwave_obj:<15.6f} {diff:<15.6f} {min(pulp_gap, dwave_gap):<10.2f} {status}")
            else:
                print(f"{n_patches:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<10} ❌ FAILED")
        else:
            print(f"{n_patches:<10} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15} {'ERROR':<10} ❌ FAILED")
    except Exception as e:
        print(f"{n_patches:<10} {'EXCEPTION':<15} {'EXCEPTION':<15} {'EXCEPTION':<15} {'ERROR':<10} ❌ {str(e)[:40]}")

print("="*120)
print("\nSUMMARY:")
print("- Objectives should now be positive and comparable")
print("- DWave may find better or worse solutions than PuLP")
print("- Gap % should be reasonable (<100% for most cases)")
print("="*120)
