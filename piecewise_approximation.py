"""
Piecewise Linear Approximation for Power Functions

This script creates a piecewise linear approximation for f(A) = A^power
with a specified number of breakpoints. This allows non-linear functions
to be used in linear programming solvers like PuLP or D-Wave CQM.

The approximation uses SOS2 (Special Ordered Set of type 2) formulation,
which is a standard technique for piecewise linear functions in MILP.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import json


class PiecewiseApproximation:
    """
    Creates and manages piecewise linear approximations for power functions.
    """
    
    def __init__(self, power: float = 0.548, num_points: int = 10, max_value: float = 100.0):
        """
        Initialize the piecewise approximation.
        
        Args:
            power: The exponent for the power function f(x) = x^power
            num_points: Number of interior points (total breakpoints = num_points + 2)
            max_value: Maximum value of x to consider
        """
        self.power = power
        self.num_points = num_points
        self.max_value = max_value
        self.breakpoints = None
        self.function_values = None
        self.slopes = None
        self.intercepts = None
        
        self._compute_approximation()
    
    def _compute_approximation(self):
        """
        Compute the piecewise linear approximation.
        
        Creates breakpoints and computes the linear segments between them.
        Total breakpoints = 2 + num_points (includes 0 and max_value endpoints)
        """
        # Create breakpoints: 0, interior points, max_value
        # Total = 2 (endpoints) + num_points (interior) breakpoints
        total_breakpoints = self.num_points + 2
        
        # Use uniform spacing for now (can be optimized later)
        self.breakpoints = np.linspace(0, self.max_value, total_breakpoints)
        
        # Compute function values at breakpoints: f(x) = x^power
        self.function_values = np.power(self.breakpoints, self.power)
        
        # Compute slopes and intercepts for each segment
        num_segments = len(self.breakpoints) - 1
        self.slopes = np.zeros(num_segments)
        self.intercepts = np.zeros(num_segments)
        
        for i in range(num_segments):
            x1, x2 = self.breakpoints[i], self.breakpoints[i + 1]
            y1, y2 = self.function_values[i], self.function_values[i + 1]
            
            # Linear segment: y = slope * x + intercept
            self.slopes[i] = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            self.intercepts[i] = y1 - self.slopes[i] * x1
    
    def evaluate(self, x: float) -> float:
        """
        Evaluate the piecewise linear approximation at point x.
        
        Args:
            x: Input value
            
        Returns:
            Approximated function value
        """
        if x < 0:
            return 0.0
        if x > self.max_value:
            # Extrapolate using last segment
            return self.slopes[-1] * x + self.intercepts[-1]
        
        # Find the segment containing x
        segment_idx = np.searchsorted(self.breakpoints[1:], x)
        
        # Evaluate linear function for this segment
        return self.slopes[segment_idx] * x + self.intercepts[segment_idx]
    
    def evaluate_true(self, x: float) -> float:
        """
        Evaluate the true power function at point x.
        
        Args:
            x: Input value
            
        Returns:
            True function value
        """
        return x ** self.power if x >= 0 else 0.0
    
    def get_max_error(self, num_test_points: int = 1000) -> Tuple[float, float, float]:
        """
        Compute the maximum approximation error over the domain.
        
        Args:
            num_test_points: Number of points to test
            
        Returns:
            Tuple of (max_absolute_error, max_relative_error, avg_absolute_error)
        """
        test_points = np.linspace(0, self.max_value, num_test_points)
        
        true_values = np.array([self.evaluate_true(x) for x in test_points])
        approx_values = np.array([self.evaluate(x) for x in test_points])
        
        abs_errors = np.abs(true_values - approx_values)
        # Avoid division by zero for relative error
        rel_errors = np.divide(abs_errors, true_values, 
                              out=np.zeros_like(abs_errors), 
                              where=true_values > 1e-10)
        
        max_abs_error = np.max(abs_errors)
        max_rel_error = np.max(rel_errors)
        avg_abs_error = np.mean(abs_errors)
        
        return max_abs_error, max_rel_error, avg_abs_error
    
    def plot(self, save_path: str = None, show: bool = True):
        """
        Plot the true function vs the piecewise approximation.
        
        Args:
            save_path: Path to save the plot (optional)
            show: Whether to display the plot
        """
        # Generate smooth curve for true function
        x_smooth = np.linspace(0, self.max_value, 1000)
        y_true = np.power(x_smooth, self.power)
        y_approx = np.array([self.evaluate(x) for x in x_smooth])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Function comparison
        ax1.plot(x_smooth, y_true, 'b-', linewidth=2, label=f'True: $x^{{{self.power}}}$')
        ax1.plot(x_smooth, y_approx, 'r--', linewidth=2, label='Piecewise Linear')
        ax1.plot(self.breakpoints, self.function_values, 'go', markersize=8, 
                label=f'Breakpoints (n={len(self.breakpoints)})')
        ax1.set_xlabel('x (Area)', fontsize=12)
        ax1.set_ylabel(f'f(x) = $x^{{{self.power}}}$', fontsize=12)
        ax1.set_title('Piecewise Linear Approximation', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error analysis
        errors = np.abs(y_true - y_approx)
        ax2.plot(x_smooth, errors, 'r-', linewidth=2)
        ax2.fill_between(x_smooth, 0, errors, alpha=0.3, color='red')
        ax2.set_xlabel('x (Area)', fontsize=12)
        ax2.set_ylabel('Absolute Error', fontsize=12)
        ax2.set_title('Approximation Error', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add error statistics
        max_abs, max_rel, avg_abs = self.get_max_error()
        error_text = f'Max Abs Error: {max_abs:.6f}\n'
        error_text += f'Max Rel Error: {max_rel*100:.2f}%\n'
        error_text += f'Avg Abs Error: {avg_abs:.6f}'
        ax2.text(0.98, 0.97, error_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def export_to_json(self, filepath: str):
        """
        Export the approximation data to JSON for use in optimization models.
        
        Args:
            filepath: Path to save JSON file
        """
        data = {
            'power': self.power,
            'num_interior_points': self.num_points,
            'total_breakpoints': len(self.breakpoints),
            'max_value': self.max_value,
            'breakpoints': self.breakpoints.tolist(),
            'function_values': self.function_values.tolist(),
            'segments': []
        }
        
        for i in range(len(self.slopes)):
            segment = {
                'index': i,
                'x_start': self.breakpoints[i],
                'x_end': self.breakpoints[i + 1],
                'y_start': self.function_values[i],
                'y_end': self.function_values[i + 1],
                'slope': self.slopes[i],
                'intercept': self.intercepts[i]
            }
            data['segments'].append(segment)
        
        # Add error statistics
        max_abs, max_rel, avg_abs = self.get_max_error()
        data['error_statistics'] = {
            'max_absolute_error': max_abs,
            'max_relative_error_percent': max_rel * 100,
            'average_absolute_error': avg_abs
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Approximation data exported to: {filepath}")
    
    def print_summary(self):
        """Print a summary of the approximation."""
        print("=" * 80)
        print("PIECEWISE LINEAR APPROXIMATION SUMMARY")
        print("=" * 80)
        print(f"Power function: f(x) = x^{self.power}")
        print(f"Domain: [0, {self.max_value}]")
        print(f"Interior points: {self.num_points}")
        print(f"Total breakpoints: {len(self.breakpoints)}")
        print(f"Number of segments: {len(self.slopes)}")
        print()
        
        print("Breakpoints:")
        for i, (x, y) in enumerate(zip(self.breakpoints, self.function_values)):
            print(f"  [{i}] x = {x:8.3f}  ->  f(x) = {y:8.3f}")
        print()
        
        print("Segments (y = slope*x + intercept):")
        for i in range(len(self.slopes)):
            print(f"  Segment {i}: [{self.breakpoints[i]:.2f}, {self.breakpoints[i+1]:.2f}]")
            print(f"    slope = {self.slopes[i]:.6f}, intercept = {self.intercepts[i]:.6f}")
        print()
        
        max_abs, max_rel, avg_abs = self.get_max_error()
        print("Error Analysis:")
        print(f"  Maximum absolute error: {max_abs:.6f}")
        print(f"  Maximum relative error: {max_rel*100:.2f}%")
        print(f"  Average absolute error: {avg_abs:.6f}")
        print("=" * 80)


def main():
    """
    Main function to demonstrate the piecewise approximation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create piecewise linear approximation for power functions'
    )
    parser.add_argument('--power', type=float, default=0.548,
                       help='Power exponent (default: 0.548)')
    parser.add_argument('--points', type=int, default=10,
                       help='Number of interior points (default: 10)')
    parser.add_argument('--max-value', type=float, default=100.0,
                       help='Maximum domain value (default: 100.0)')
    parser.add_argument('--output', type=str, default='piecewise_approx.json',
                       help='Output JSON filename (default: piecewise_approx.json)')
    parser.add_argument('--plot', type=str, default='piecewise_approx.png',
                       help='Output plot filename (default: piecewise_approx.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t display the plot')
    
    args = parser.parse_args()
    
    # Create approximation
    print(f"\nCreating piecewise linear approximation for f(x) = x^{args.power}")
    print(f"Using {args.points} interior points (total {args.points + 2} breakpoints)")
    print(f"Domain: [0, {args.max_value}]\n")
    
    approx = PiecewiseApproximation(
        power=args.power,
        num_points=args.points,
        max_value=args.max_value
    )
    
    # Print summary
    approx.print_summary()
    
    # Export to JSON
    approx.export_to_json(args.output)
    
    # Create plot
    print(f"\nGenerating plot...")
    approx.plot(save_path=args.plot, show=not args.no_show)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
