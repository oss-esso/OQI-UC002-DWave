"""
Clique QPU Solver - Add this to qpu_benchmark.py after solve_direct_qpu
"""

def solve_clique_qpu(cqm: ConstrainedQuadraticModel, data: Dict,
                     num_reads: int = 1000, annealing_time: int = 20) -> Dict:
    """
    Clique QPU: CQM → BQM → DWaveCliqueSampler (zero embedding overhead)
    
    Uses DWaveCliqueSampler which maps directly to hardware cliques (n<=16).
    **Critical**: Problem must be small enough to fit in a clique!
    """
    if not HAS_CLIQUE:
        return {'success': False, 'error': 'DWaveCliqueSampler not available'}
    
    result = {
        'method': 'clique_qpu',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'timings': {},
    }
    
    total_start = time.time()
    lagrange_multiplier = 50.0  # Same as direct QPU for fair comparison
    
    try:
        # Convert CQM to BQM
        LOG.info(f"  [CliqueQPU] Converting CQM to BQM (lagrange={lagrange_multiplier:.1f})...")
        t0 = time.time()
        bqm, info = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
        result['timings']['cqm_to_bqm'] = time.time() - t0
        result['bqm_variables'] = len(bqm.variables)
        result['bqm_interactions'] = len(bqm.quadratic)
        result['lagrange_multiplier'] = lagrange_multiplier
        
        LOG.info(f"  [CliqueQPU] BQM: {result['bqm_variables']} vars, {result['bqm_interactions']} interactions")
        
        # Check if problem fits in clique (typically n<=16 for Pegasus)
        n_vars = len(bqm.variables)
        if n_vars > 20:
            LOG.warning(f"  [CliqueQPU] Problem too large ({n_vars} vars) for guaranteed clique embedding")
            LOG.warning(f"  [CliqueQPU] DWaveCliqueSampler works best for n<=16, may use chains for n>16")
        
        # Sample with DWaveCliqueSampler (automatic clique detection + embedding)
        LOG.info(f"  [CliqueQPU] Sampling with DWaveCliqueSampler ({num_reads} reads)...")
        
        sampler = DWaveCliqueSampler()
        
        sample_start = time.time()
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            annealing_time=annealing_time,
            label=f"CliqueQPU_{data['n_farms']}farms"
        )
        result['timings']['sampling'] = time.time() - sample_start
        
        # Extract timing info
        timing_info = sampleset.info.get('timing', {})
        qpu_access_us = timing_info.get('qpu_access_time', 0)
        qpu_programming_us = timing_info.get('qpu_programming_time', 0)
        qpu_sampling_us = timing_info.get('qpu_sampling_time', 0)
        total_real_us = timing_info.get('total_real_time', 0)
        
        result['timings']['qpu_access'] = qpu_access_us / 1e6
        result['timings']['qpu_programming'] = qpu_programming_us / 1e6
        result['timings']['qpu_sampling'] = qpu_sampling_us / 1e6
        result['timings']['qpu_total_real'] = total_real_us / 1e6
        
        # Embedding info (cliques have minimal embedding overhead)
        embedding_info = sampleset.info.get('embedding_context', {})
        embedding_time = embedding_info.get('embedding_time', 0)
        chain_strength_used = embedding_info.get('chain_strength', 0)
        
        result['timings']['embedding'] = embedding_time / 1e6 if embedding_time else 0
        result['timings']['embedding_total'] = result['timings']['embedding']
        result['timings']['qpu_access_total'] = result['timings']['qpu_access']
        result['timings']['solve_time'] = result['timings']['qpu_access'] + result['timings']['embedding']
        result['chain_strength'] = chain_strength_used
        
        # Check for chain breaks (should be minimal/zero for cliques)
        if hasattr(sampleset.record, 'chain_break_fraction'):
            result['chain_break_fraction'] = float(np.mean(sampleset.record.chain_break_fraction))
        else:
            result['chain_break_fraction'] = 0.0
        
        # Extract embedding details if available
        if 'embedding' in embedding_info:
            embedding = embedding_info['embedding']
            result['physical_qubits'] = sum(len(chain) for chain in embedding.values())
            result['max_chain_length'] = max(len(chain) for chain in embedding.values()) if embedding else 1
            LOG.info(f"  [CliqueQPU] Embedding: {result['physical_qubits']} physical qubits, "
                    f"max chain {result['max_chain_length']}")
        else:
            result['physical_qubits'] = n_vars  # Assume 1:1 for cliques
            result['max_chain_length'] = 1
        
        # Best solution
        best = sampleset.first
        result['best_energy'] = float(best.energy)
        result['n_samples'] = len(sampleset)
        
        # Extract and evaluate solution
        result['solution'] = extract_solution(best.sample, data)
        result['objective'] = calculate_objective(best.sample, data)
        result['violations'] = count_violations(best.sample, data)
        result['violation_details'] = get_detailed_violations(best.sample, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        
        LOG.info(f"  [CliqueQPU] Success! obj={result['objective']:.4f}, "
                f"QPU_access={result['timings']['qpu_access']:.3f}s, "
                f"embed={result['timings']['embedding']:.3f}s, "
                f"chains={result['chain_break_fraction']:.2%}")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        LOG.error(f"  [CliqueQPU] Error: {e}")
        import traceback
        traceback.print_exc()
    
    return result
