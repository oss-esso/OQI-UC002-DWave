#!/usr/bin/env python3
"""
Minimal QPU Validation Test - Pure Quantum (No Hybrid Solvers)

This script validates D-Wave access before running the full benchmark.
Tests ONLY:
1. Direct QPU (DWaveSampler)
2. QBSolv
3. Embedding tools (minorminer)

NO hybrid solvers are tested or used.
"""

import sys
import os

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main():
    print("=" * 70)
    print("QPU BENCHMARK: Validation Test (Pure Quantum - No Hybrid)")
    print("=" * 70)
    print()
    
    results = {
        'qpu_direct': False,
        'qbsolv': False,
        'embedding': False,
        'neal': False,
    }
    
    # ========================================================================
    # 1. Check Direct QPU Access (DWaveSampler)
    # ========================================================================
    print("[1/4] Checking Direct QPU Access (DWaveSampler)...")
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
        sampler = DWaveSampler()
        chip_id = sampler.properties.get('chip_id', 'Unknown')
        num_qubits = sampler.properties.get('num_qubits', 0)
        topology = sampler.properties.get('topology', {}).get('type', 'Unknown')
        
        print(f"  ✅ QPU Available: {chip_id}")
        print(f"     Qubits: {num_qubits}")
        print(f"     Topology: {topology}")
        results['qpu_direct'] = True
        results['qpu_info'] = {
            'chip_id': chip_id,
            'num_qubits': num_qubits,
            'topology': topology
        }
    except Exception as e:
        print(f"  ❌ Direct QPU not available: {e}")
        results['qpu_error'] = str(e)
    
    print()
    
    # ========================================================================
    # 2. Check QBSolv
    # ========================================================================
    print("[2/4] Checking QBSolv...")
    try:
        from dwave_qbsolv import QBSolv
        print(f"  ✅ QBSolv Available")
        results['qbsolv'] = True
    except ImportError:
        try:
            from qbsolv import QBSolv
            print(f"  ✅ QBSolv Available (alt import)")
            results['qbsolv'] = True
        except ImportError as e:
            print(f"  ❌ QBSolv not available: {e}")
            print(f"     Install with: pip install dwave-qbsolv")
            results['qbsolv_error'] = str(e)
    
    print()
    
    # ========================================================================
    # 3. Check Embedding Tools (minorminer)
    # ========================================================================
    print("[3/4] Checking Embedding Tools (minorminer)...")
    try:
        from minorminer import find_embedding
        import dwave_networkx as dnx
        
        # Quick test: embed a tiny problem
        import networkx as nx
        source = nx.complete_graph(4)
        target = dnx.pegasus_graph(2)  # Small Pegasus for testing
        
        embedding = find_embedding(source, target, timeout=5)
        if embedding:
            print(f"  ✅ minorminer Available")
            print(f"     Test embedding: K4 → P2 succeeded")
            results['embedding'] = True
        else:
            print(f"  ⚠️  minorminer installed but embedding failed")
            results['embedding'] = False
    except ImportError as e:
        print(f"  ❌ minorminer not available: {e}")
        results['embedding_error'] = str(e)
    except Exception as e:
        print(f"  ⚠️  minorminer test error: {e}")
        results['embedding_error'] = str(e)
    
    print()
    
    # ========================================================================
    # 4. Check Simulated Annealing (neal) for fallback
    # ========================================================================
    print("[4/4] Checking Simulated Annealing (neal) for fallback...")
    try:
        import neal
        from dimod import BinaryQuadraticModel
        
        # Quick test
        bqm = BinaryQuadraticModel({'a': -1, 'b': -1}, {('a', 'b'): 2}, 0, 'BINARY')
        sampler = neal.SimulatedAnnealingSampler()
        result = sampler.sample(bqm, num_reads=10)
        
        print(f"  ✅ neal (SimulatedAnnealing) Available")
        print(f"     Can be used as fallback when QPU unavailable")
        results['neal'] = True
    except ImportError as e:
        print(f"  ❌ neal not available: {e}")
        results['neal_error'] = str(e)
    except Exception as e:
        print(f"  ⚠️  neal test error: {e}")
        results['neal_error'] = str(e)
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY (Pure QPU - No Hybrid)")
    print("=" * 70)
    
    qpu_ready = results['qpu_direct']
    fallback_ready = results['neal'] or results['qbsolv']
    
    print(f"  Direct QPU:      {'✅ Ready' if results['qpu_direct'] else '❌ Not available'}")
    print(f"  QBSolv:          {'✅ Ready' if results['qbsolv'] else '❌ Not available'}")
    print(f"  Embedding:       {'✅ Ready' if results['embedding'] else '❌ Not available'}")
    print(f"  SimulatedAnneal: {'✅ Ready' if results['neal'] else '❌ Not available'}")
    
    print()
    
    if qpu_ready:
        print("✅ QPU READY - Can run full benchmark on quantum hardware!")
        print()
        print("Next steps:")
        print("  1. Run: python qpu_benchmark.py --test 25  (quick 25-farm test)")
        print("  2. Run: python qpu_benchmark.py --full     (full benchmark)")
    elif fallback_ready:
        print("⚠️  QPU NOT AVAILABLE - Can run benchmark with SimulatedAnnealing fallback")
        print()
        print("To run with fallback:")
        print("  python qpu_benchmark.py --test 25 --no-qpu")
        print()
        print("To enable QPU:")
        print("  1. Set DWAVE_API_TOKEN environment variable")
        print("  2. Run: dwave config create")
        print("  3. Visit: https://cloud.dwavesys.com/leap/")
    else:
        print("❌ Cannot run benchmark - no solvers available")
        print()
        print("Install dependencies:")
        print("  pip install dwave-ocean-sdk neal dwave-qbsolv")
    
    print()
    
    # Save results to JSON for reference
    import json
    from datetime import datetime
    
    results['timestamp'] = datetime.now().isoformat()
    results['qpu_ready'] = qpu_ready
    results['fallback_ready'] = fallback_ready
    
    output_file = os.path.join(os.path.dirname(__file__), 'qpu_validation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")
    
    return 0 if (qpu_ready or fallback_ready) else 1


if __name__ == "__main__":
    sys.exit(main())
