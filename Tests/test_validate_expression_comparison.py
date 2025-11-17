"""
Test script to verify constraint expression comparison logic.

This tests that the validator correctly compares LHS-RHS expressions
regardless of term ordering.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.validate_cqm_vs_pulp import CQMPuLPValidator


def test_expression_comparison():
    """Test the expression comparison method."""
    
    # Create a mock validator instance (we only need the methods)
    class MockValidator(CQMPuLPValidator):
        def __init__(self):
            # Skip parent __init__, we only need the comparison methods
            pass
    
    validator = MockValidator()
    
    print("="*80)
    print("TESTING EXPRESSION COMPARISON")
    print("="*80)
    
    # Test 1: Identical expressions
    print("\nTest 1: Identical expressions")
    expr1 = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0, '__CONSTANT__': -5.0}
    expr2 = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0, '__CONSTANT__': -5.0}
    match, diffs = validator._compare_constraint_expressions(expr1, expr2)
    print(f"  Match: {match} (expected: True)")
    assert match, "Should match"
    
    # Test 2: Same expression, different term order (both dicts should compare equal)
    print("\nTest 2: Same expression, different order")
    expr1 = {'x1': 1.0, 'x2': 2.0, '__CONSTANT__': -5.0}
    expr2 = {'__CONSTANT__': -5.0, 'x2': 2.0, 'x1': 1.0}
    match, diffs = validator._compare_constraint_expressions(expr1, expr2)
    print(f"  Match: {match} (expected: True)")
    assert match, "Should match regardless of order"
    
    # Test 3: Different coefficients
    print("\nTest 3: Different coefficients")
    expr1 = {'x1': 1.0, 'x2': 2.0, '__CONSTANT__': -5.0}
    expr2 = {'x1': 1.0, 'x2': 3.0, '__CONSTANT__': -5.0}
    match, diffs = validator._compare_constraint_expressions(expr1, expr2)
    print(f"  Match: {match} (expected: False)")
    print(f"  Differences: {diffs}")
    assert not match, "Should not match"
    assert len(diffs) == 1, "Should have 1 difference"
    
    # Test 4: Missing variables (treated as 0)
    print("\nTest 4: Missing variables")
    expr1 = {'x1': 1.0, 'x2': 2.0, '__CONSTANT__': -5.0}
    expr2 = {'x1': 1.0, '__CONSTANT__': -5.0}  # x2 missing
    match, diffs = validator._compare_constraint_expressions(expr1, expr2)
    print(f"  Match: {match} (expected: False)")
    print(f"  Differences: {diffs}")
    assert not match, "Should not match when variable is missing"
    
    # Test 5: Extra zero-coefficient variables should match
    print("\nTest 5: Explicit zero coefficients")
    expr1 = {'x1': 1.0, 'x2': 2.0}
    expr2 = {'x1': 1.0, 'x2': 2.0, 'x3': 0.0}
    match, diffs = validator._compare_constraint_expressions(expr1, expr2)
    print(f"  Match: {match} (expected: True)")
    assert match, "Should match when extra variable has zero coefficient"
    
    # Test 6: Numerical tolerance test
    print("\nTest 6: Numerical tolerance")
    expr1 = {'x1': 1.0, 'x2': 2.0}
    expr2 = {'x1': 1.0000001, 'x2': 2.0000001}
    match, diffs = validator._compare_constraint_expressions(expr1, expr2, tolerance=1e-6)
    print(f"  Match: {match} (expected: True with tolerance=1e-6)")
    assert match, "Should match within tolerance"
    
    # Test 7: Different constants (RHS values)
    print("\nTest 7: Different RHS constants")
    expr1 = {'x1': 1.0, 'x2': 2.0, '__CONSTANT__': -5.0}
    expr2 = {'x1': 1.0, 'x2': 2.0, '__CONSTANT__': -10.0}
    match, diffs = validator._compare_constraint_expressions(expr1, expr2)
    print(f"  Match: {match} (expected: False)")
    print(f"  Differences: {diffs}")
    assert not match, "Should not match with different constants"
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    test_expression_comparison()
