"""
Wrapper for clique_decomposition to handle dependencies and logging.
"""
import logging
import sys
from typing import Dict
from pathlib import Path

# Setup logging for clique_decomposition module
LOG = logging.getLogger('clique_decomp')
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    LOG.addHandler(handler)

# Make LOG available globally for clique_decomposition import
import sys
sys.modules['__main__'].LOG = LOG

# Now import clique decomposition
try:
    from clique_decomposition import solve_rotation_clique_decomposition as _solve_clique
    HAS_CLIQUE = True
except ImportError as e:
    HAS_CLIQUE = False
    _solve_clique = None
    print(f"Warning: Could not import clique_decomposition: {e}")

def solve_clique_wrapper(data: Dict, cqm, num_reads: int = 100) -> Dict:
    """
    Wrapper for solve_rotation_clique_decomposition that handles missing dependencies.
    """
    if not HAS_CLIQUE or _solve_clique is None:
        return {
            'method': 'clique_decomposition',
            'status': 'error',
            'objective': None,
            'solution': None,
            'error': 'Clique decomposition not available'
        }
    
    try:
        result = _solve_clique(data, cqm, num_reads)
        
        # Normalize result format
        if 'objective' in result and result['objective'] is not None:
            return {
                'method': 'clique_decomposition',
                'status': 'success',
                'objective': result['objective'],
                'solution': result.get('solution', {}),
                'runtime': result.get('total_time', result.get('wall_time', 0)),
                'qpu_time': result.get('timings', {}).get('qpu_access_total', 0),
                'violations': result.get('violations', 0),
                'raw_result': result,
            }
        else:
            return {
                'method': 'clique_decomposition',
                'status': 'failed',
                'objective': None,
                'solution': None,
                'error': 'No valid solution found'
            }
    
    except Exception as e:
        import traceback
        return {
            'method': 'clique_decomposition',
            'status': 'error',
            'objective': None,
            'solution': None,
            'error': f'{type(e).__name__}: {str(e)}',
            'traceback': traceback.format_exc()
        }
