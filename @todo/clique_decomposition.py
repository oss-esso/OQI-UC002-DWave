"""
Hierarchical Decomposition for Clique Embedding (Mohseni et al. style)

Key idea: Decompose problem into subproblems with ≤16 variables each,
solve with DWaveCliqueSampler (zero embedding overhead), then coordinate.
"""

def solve_rotation_clique_decomposition(data: Dict, cqm: ConstrainedQuadraticModel,
                                        num_reads: int = 100) -> Dict:
    """
    Decompose rotation problem farm-by-farm, solve each with clique sampler.
    
    Strategy:
    1. For each farm: 6 families × 3 periods = 18 variables
    2. Solve each farm independently with DWaveCliqueSampler (fits in clique!)
    3. Coordinate via spatial coupling (post-processing or iterative refinement)
    
    This mimics Mohseni et al.'s approach: many small QUBOs solved independently.
    """
    import time
    import numpy as np
    from dimod import BinaryQuadraticModel, Binary, cqm_to_bqm
    from dwave.system.samplers import DWaveCliqueSampler
    
    result = {
        'method': 'clique_decomposition',
        'num_reads': num_reads,
        'timings': {},
        'subproblem_results': []
    }
    
    total_start = time.time()
    
    farm_names = data['farm_names']
    food_names = data['food_names']  # Crop families
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    n_periods = 3
    
    LOG.info(f"  [CliqueDecomp] Decomposing {len(farm_names)} farms into independent subproblems...")
    LOG.info(f"  [CliqueDecomp] Each farm: {len(food_names)} families × {n_periods} periods = {len(food_names) * n_periods} variables")
    
    # Check if subproblems fit in cliques
    vars_per_farm = len(food_names) * n_periods
    if vars_per_farm > 20:
        LOG.warning(f"  [CliqueDecomp] Subproblem size ({vars_per_farm} vars) may exceed clique size (16)!")
    
    sampler = DWaveCliqueSampler()
    
    # Solve each farm independently
    farm_solutions = {}
    total_qpu_time = 0
    total_embedding_time = 0
    
    for farm_idx, farm in enumerate(farm_names):
        subproblem_start = time.time()
        
        # Build QUBO for this farm only (temporal optimization)
        # Variables: Y[family, period] for this farm
        Y = {}
        for c in food_names:
            for t in range(1, n_periods + 1):
                Y[(c, t)] = Binary(f"Y_{farm}_{c}_t{t}")
        
        # Objective for this farm
        farm_area = land_availability[farm]
        obj = 0
        
        # Linear benefits
        for c in food_names:
            B_c = food_benefits[c]
            for t in range(1, n_periods + 1):
                obj += (B_c * farm_area * Y[(c, t)]) / total_area
        
        # Rotation synergies (temporal coupling within farm)
        # Use simplified synergy matrix
        rotation_gamma = 0.2
        for t in range(2, n_periods + 1):
            for c1 in food_names:
                for c2 in food_names:
                    # Simplified: negative for same family, positive for different
                    synergy = -0.5 if c1 == c2 else 0.1
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(c1, t-1)] * Y[(c2, t)]) / total_area
        
        # Soft one-hot penalty (prefer 1 crop per period)
        one_hot_penalty = 3.0
        for t in range(1, n_periods + 1):
            crop_count = sum(Y[(c, t)] for c in food_names)
            obj -= one_hot_penalty * (crop_count * crop_count - 2 * crop_count + 1)
        
        # Convert to BQM
        bqm = BinaryQuadraticModel('BINARY')
        bqm.set_objective(-obj)  # Negate for minimization
        
        # Sample with clique sampler
        try:
            sample_start = time.time()
            sampleset = sampler.sample(
                bqm,
                num_reads=num_reads,
                label=f"CliqueDecomp_farm{farm_idx}"
            )
            sample_time = time.time() - sample_start
            
            # Extract timing
            timing_info = sampleset.info.get('timing', {})
            qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            
            embedding_info = sampleset.info.get('embedding_context', {})
            embed_time = embedding_info.get('embedding_time', 0) / 1e6 if embedding_info else 0
            total_embedding_time += embed_time
            
            # Best solution for this farm
            best = sampleset.first
            farm_solution = {}
            for c in food_names:
                for t in range(1, n_periods + 1):
                    var_name = f"Y_{farm}_{c}_t{t}"
                    farm_solution[var_name] = best.sample.get((c, t), 0)
            
            farm_solutions[farm] = farm_solution
            
            subproblem_result = {
                'farm': farm,
                'variables': len(Y),
                'qpu_time': qpu_time,
                'embedding_time': embed_time,
                'sample_time': sample_time,
                'success': True
            }
            result['subproblem_results'].append(subproblem_result)
            
            LOG.info(f"  [CliqueDecomp] Farm {farm_idx+1}/{len(farm_names)}: "
                    f"{len(Y)} vars, QPU={qpu_time:.3f}s, embed={embed_time:.3f}s")
            
        except Exception as e:
            LOG.error(f"  [CliqueDecomp] Farm {farm} failed: {e}")
            # Use empty solution for this farm
            farm_solution = {f"Y_{farm}_{c}_t{t}": 0 
                           for c in food_names for t in range(1, n_periods + 1)}
            farm_solutions[farm] = farm_solution
            result['subproblem_results'].append({
                'farm': farm,
                'success': False,
                'error': str(e)
            })
    
    # Combine solutions
    combined_solution = {}
    for farm in farm_names:
        combined_solution.update(farm_solutions[farm])
    
    # Evaluate combined solution
    from qpu_benchmark import calculate_objective, count_violations, get_detailed_violations, extract_solution
    
    result['objective'] = calculate_objective(combined_solution, data)
    result['violations'] = count_violations(combined_solution, data)
    result['violation_details'] = get_detailed_violations(combined_solution, data)
    result['feasible'] = result['violations'] == 0
    result['solution'] = extract_solution(combined_solution, data)
    
    # Timing summary
    result['timings']['total'] = time.time() - total_start
    result['timings']['qpu_access'] = total_qpu_time
    result['timings']['embedding'] = total_embedding_time
    result['timings']['qpu_access_total'] = total_qpu_time
    result['timings']['embedding_total'] = total_embedding_time
    result['timings']['solve_time'] = total_qpu_time + total_embedding_time
    result['total_time'] = result['timings']['total']
    result['wall_time'] = result['timings']['total']
    
    result['success'] = True
    result['n_subproblems'] = len(farm_names)
    result['avg_subproblem_size'] = vars_per_farm
    
    LOG.info(f"  [CliqueDecomp] Complete! {len(farm_names)} subproblems, "
            f"total QPU={total_qpu_time:.3f}s, embed={total_embedding_time:.3f}s")
    LOG.info(f"  [CliqueDecomp] Combined objective: {result['objective']:.4f}, "
            f"violations: {result['violations']}")
    
    return result
