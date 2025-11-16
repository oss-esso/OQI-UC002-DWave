"""
8-SAT Instance Generation Example

Demonstrates k-SAT generation with k=8 (much easier than 3-SAT).
Higher k values make satisfiability easier - phase transition moves to much higher α.
"""

from qaoa_sat_instance import generate_random_ksat, generate_planted_ksat
from hardness_metrics import compute_hardness_metrics

def main():
    """Generate and analyze 8-SAT instances"""
    print("="*70)
    print("8-SAT INSTANCE GENERATION EXAMPLE")
    print("="*70)
    
    print("\nNote: k-SAT phase transition shifts with k:")
    print("  k=3: α ≈ 4.27 (hard)")
    print("  k=4: α ≈ 9.93")
    print("  k=5: α ≈ 21.12")
    print("  k=8: α ≈ 87 (theoretical)")
    print("  Higher k → easier to satisfy, need more clauses for hardness\n")
    
    # Example 1: Small 8-SAT
    print("1. Small 8-SAT Instance (n=30, α=2.0)")
    print("-" * 70)
    sat8_small = generate_random_ksat(n=30, k=8, alpha=2.0, seed=42)
    print(f"   Variables: {sat8_small.n}")
    print(f"   Clauses: {sat8_small.m}")
    print(f"   k-value: {sat8_small.k}")
    print(f"   α (m/n): {sat8_small.alpha:.3f}")
    print(f"   First clause (8 literals): {sat8_small.clauses[0]}")
    print(f"   Clause length verification: {len(sat8_small.clauses[0])} literals")
    
    # Check satisfiability
    satisfied = 0
    for clause in sat8_small.clauses:
        # Random assignment (all False)
        if any(lit < 0 for lit in clause):
            satisfied += 1
    print(f"   Clauses satisfied by all-False: {satisfied}/{sat8_small.m}")
    print(f"   Satisfaction rate: {satisfied/sat8_small.m*100:.1f}%")
    
    # Example 2: Medium 8-SAT
    print("\n2. Medium 8-SAT Instance (n=50, α=5.0)")
    print("-" * 70)
    sat8_medium = generate_random_ksat(n=50, k=8, alpha=5.0, seed=123)
    print(f"   Variables: {sat8_medium.n}")
    print(f"   Clauses: {sat8_medium.m}")
    print(f"   α (m/n): {sat8_medium.alpha:.3f}")
    print(f"   Sample clauses:")
    for i in range(3):
        print(f"     Clause {i+1}: {sat8_medium.clauses[i]}")
    
    # Example 3: Planted 8-SAT
    print("\n3. Planted 8-SAT Instance (n=40, α=3.0)")
    print("-" * 70)
    sat8_planted = generate_planted_ksat(n=40, k=8, alpha=3.0, seed=456)
    print(f"   Variables: {sat8_planted.n}")
    print(f"   Clauses: {sat8_planted.m}")
    print(f"   Planted solution: {sat8_planted.planted_solution[:15]}...")
    
    # Verify planted solution works
    satisfied_planted = 0
    for clause in sat8_planted.clauses:
        clause_sat = False
        for lit in clause:
            var = abs(lit) - 1
            if var < len(sat8_planted.planted_solution):
                val = sat8_planted.planted_solution[var]
                if (lit > 0 and val) or (lit < 0 and not val):
                    clause_sat = True
                    break
        if clause_sat:
            satisfied_planted += 1
    print(f"   Planted solution satisfies: {satisfied_planted}/{sat8_planted.m} clauses")
    print(f"   Success rate: {satisfied_planted/sat8_planted.m*100:.1f}%")
    
    # Example 4: Hardness comparison
    print("\n4. Hardness Comparison: 3-SAT vs 8-SAT")
    print("-" * 70)
    
    # Generate comparable 3-SAT
    sat3 = generate_random_ksat(n=30, k=3, alpha=4.27, seed=42)
    metrics3 = compute_hardness_metrics(sat3.n, sat3.clauses)
    
    # Metrics for 8-SAT
    metrics8 = compute_hardness_metrics(sat8_small.n, sat8_small.clauses)
    
    print(f"   3-SAT (n=30, α=4.27):")
    print(f"     Hardness Score: {metrics3.hardness_score:.1f}/100")
    print(f"     Difficulty: {metrics3.expected_difficulty}")
    print(f"     VCG Density: {metrics3.vcg_density:.4f}")
    
    print(f"\n   8-SAT (n=30, α=2.0):")
    print(f"     Hardness Score: {metrics8.hardness_score:.1f}/100")
    print(f"     Difficulty: {metrics8.expected_difficulty}")
    print(f"     VCG Density: {metrics8.vcg_density:.4f}")
    
    print(f"\n   Key Insight: 8-SAT is easier despite larger clause size")
    print(f"   because each clause has many ways to be satisfied.")
    
    # Example 5: Export to files
    print("\n5. Export to DIMACS CNF Format")
    print("-" * 70)
    
    with open("8sat_small_n30.cnf", 'w') as f:
        f.write(sat8_small.to_dimacs_cnf())
    print(f"   ✓ Saved: 8sat_small_n30.cnf")
    
    with open("8sat_medium_n50.cnf", 'w') as f:
        f.write(sat8_medium.to_dimacs_cnf())
    print(f"   ✓ Saved: 8sat_medium_n50.cnf")
    
    with open("8sat_planted_n40.cnf", 'w') as f:
        f.write(sat8_planted.to_dimacs_cnf())
    print(f"   ✓ Saved: 8sat_planted_n40.cnf")
    
    # Show file preview
    print("\n6. DIMACS File Preview (8sat_small_n30.cnf)")
    print("-" * 70)
    with open("8sat_small_n30.cnf", 'r') as f:
        lines = f.readlines()[:15]
        for line in lines:
            print(f"   {line.rstrip()}")
    print(f"   ... ({sat8_small.m - 13} more clauses)")
    
    print("\n" + "="*70)
    print("✓ 8-SAT EXAMPLES COMPLETED")
    print("="*70)
    print("\nUse Cases for 8-SAT:")
    print("  • Easier baseline for algorithm testing")
    print("  • Scalability studies (can generate large satisfiable instances)")
    print("  • Quantum algorithm warmup (before tackling hard 3-SAT)")
    print("  • Circuit depth analysis (more qubits per clause)")
    print("\nNote: For QAOA hardness testing, use 3-SAT at α≈4.27")
    print("="*70)


if __name__ == "__main__":
    main()
