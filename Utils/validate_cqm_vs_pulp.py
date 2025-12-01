"""
Validate CQM constraints against PuLP formulation.

This utility compares CQM constraint formulation with PuLP (Gurobi) constraints
to ensure they match before submitting expensive D-Wave solver jobs.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


class CQMPuLPValidator:
    """Validates CQM constraints against PuLP model."""
    
    def __init__(self, cqm, pulp_model, scenario_type: str, scenario_info: Dict):
        """
        Initialize validator.
        
        Args:
            cqm: ConstrainedQuadraticModel object
            pulp_model: PuLP model object
            scenario_type: 'farm' or 'patch'
            scenario_info: Dictionary with scenario metadata
        """
        self.cqm = cqm
        self.pulp = pulp_model
        self.scenario_type = scenario_type
        self.info = scenario_info
        self.discrepancies = []
        self.warnings = []
    
    def _normalize_variable_name(self, var_name: str) -> str:
        """
        Normalize variable names to handle PuLP tuple-based names vs CQM string names.
        
        PuLP creates variables with tuple keys like ('Patch1', 'Tomatoes'), resulting in
        names like "X_('Patch1',_'Tomatoes')". CQM uses string names like "Y_Patch1_Tomatoes".
        
        This function normalizes both formats to a common format for comparison.
        
        Examples:
            "X_('Patch1',_'Tomatoes')" -> "Patch1_Tomatoes"
            "Y_Patch1_Tomatoes" -> "Patch1_Tomatoes"
            "U_Spinach" -> "Spinach"
        
        Returns:
            Normalized variable name without prefix and with underscores
        """
        import re
        
        # Remove common prefixes (Y_, X_, U_, A_, etc.)
        name = re.sub(r'^[A-Z]_', '', var_name)
        
        # Handle PuLP tuple format: ('Patch1',_'Tomatoes') -> Patch1_Tomatoes
        # Remove parens, quotes, and normalize separators
        name = name.replace("('", "").replace("')", "").replace("',_'", "_").replace("', '", "_")
        name = name.replace("(", "").replace(")", "").replace("'", "").replace(", ", "_")
        
        return name
    
    def _extract_pulp_constraint_expression(self, constraint) -> Dict[str, float]:
        """
        Extract variable coefficients from PuLP constraint as LHS - RHS.
        
        Returns:
            Dict mapping variable names to coefficients in normalized form (LHS - RHS)
        """
        coeffs = {}
        
        try:
            # PuLP constraints have form: expression {<=, ==, >=} constant
            # We want to normalize to: LHS - RHS = 0 (or <= 0, >= 0)
            
            # Extract LHS terms
            if hasattr(constraint, 'toDict'):
                lhs_dict = constraint.toDict()
                for var, coeff in lhs_dict.items():
                    if var != 'constant':
                        var_name = str(var.name) if hasattr(var, 'name') else str(var)
                        normalized_name = self._normalize_variable_name(var_name)
                        coeffs[normalized_name] = float(coeff)
                
                # Subtract RHS constant (move to LHS)
                # PuLP stores as: sum(coeffs) + constant {sense} 0
                # We want: sum(coeffs) - rhs_value
                if 'constant' in lhs_dict:
                    rhs_value = -float(lhs_dict['constant'])
                else:
                    rhs_value = 0.0
                    
            else:
                # Alternative extraction method
                # Get the constraint expression
                expr = constraint
                
                # Try to access variables and coefficients
                if hasattr(expr, 'items'):
                    for var, coeff in expr.items():
                        if hasattr(var, 'name'):
                            normalized_name = self._normalize_variable_name(var.name)
                            coeffs[normalized_name] = float(coeff)
                
                # Get RHS
                rhs_value = float(getattr(constraint, 'constant', 0.0))
            
            # Add negative RHS as a constant term (to represent LHS - RHS)
            if abs(rhs_value) > 1e-10:
                coeffs['__CONSTANT__'] = -rhs_value
                
        except Exception as e:
            # If extraction fails, return empty dict
            print(f"    Warning: Could not extract PuLP constraint expression: {e}")
            return {}
        
        return coeffs
    
    def _extract_cqm_constraint_expression(self, constraint) -> Dict[str, float]:
        """
        Extract variable coefficients from CQM constraint as LHS - RHS.
        
        CQM stores constraints in form: LHS {<=, ==, >=} RHS
        We normalize to: LHS - RHS
        
        Returns:
            Dict mapping variable names to coefficients in normalized form (LHS - RHS)
        """
        coeffs = {}
        
        try:
            lhs = constraint.lhs
            rhs = constraint.rhs
            
            # Extract linear terms from LHS
            if hasattr(lhs, 'linear'):
                for var, coeff in lhs.linear.items():
                    normalized_name = self._normalize_variable_name(str(var))
                    coeffs[normalized_name] = float(coeff)
            
            # Extract quadratic terms (if any)
            if hasattr(lhs, 'quadratic'):
                for (var1, var2), coeff in lhs.quadratic.items():
                    quad_key = f"{var1}*{var2}"
                    coeffs[quad_key] = float(coeff)
            
            # Extract offset from LHS
            if hasattr(lhs, 'offset'):
                lhs_offset = float(lhs.offset)
            else:
                lhs_offset = 0.0
            
            # Normalize: LHS - RHS
            # Add the constant term as (LHS_offset - RHS)
            constant_term = lhs_offset - float(rhs)
            if abs(constant_term) > 1e-10:
                coeffs['__CONSTANT__'] = constant_term
                
        except Exception as e:
            print(f"    Warning: Could not extract CQM constraint expression: {e}")
            return {}
        
        return coeffs
    
    def _compare_constraint_expressions(
        self, 
        pulp_coeffs: Dict[str, float], 
        cqm_coeffs: Dict[str, float],
        tolerance: float = 1e-6
    ) -> Tuple[bool, List[str]]:
        """
        Compare two constraint expressions (as coefficient dictionaries).
        
        Returns:
            Tuple of (expressions_match, list_of_differences)
        """
        differences = []
        
        # Get all variable names from both
        all_vars = set(pulp_coeffs.keys()) | set(cqm_coeffs.keys())
        
        for var in all_vars:
            pulp_val = pulp_coeffs.get(var, 0.0)
            cqm_val = cqm_coeffs.get(var, 0.0)
            
            # Compare with tolerance
            if abs(pulp_val - cqm_val) > tolerance:
                differences.append(
                    f"Variable '{var}': PuLP={pulp_val:.6f}, CQM={cqm_val:.6f}, "
                    f"diff={abs(pulp_val - cqm_val):.6e}"
                )
        
        return len(differences) == 0, differences
    
    def _extract_plot_id_from_name(self, name: str) -> str:
        """
        Extract plot/unit identifier from constraint name.
        
        Examples:
            "Max_Assignment_Plot_5" -> "5"
            "AtMostOne_Plot_12" -> "12"
            "MaxArea_3" -> "3"
        """
        import re
        # Try to find a number in the name
        numbers = re.findall(r'\d+', name)
        if numbers:
            return numbers[-1]  # Return last number found
        return name  # Return full name if no number found


        
    def validate(self) -> Tuple[bool, List[Dict], List[Dict]]:
        """
        Run validation comparing CQM to PuLP.
        
        Returns:
            Tuple of (is_valid, discrepancies, warnings)
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING CQM vs PuLP - {self.scenario_type.upper()} SCENARIO")
        print(f"{'='*80}")
        
        # Get constraint counts
        pulp_constraints = [c for c in self.pulp.constraints.values()]
        cqm_constraints = list(self.cqm.constraints.items())
        
        print(f"PuLP constraints: {len(pulp_constraints)}")
        print(f"CQM constraints: {len(cqm_constraints)}")
        
        # Run specific validations
        self.check_constraint_count()
        self.check_variable_count()
        self.check_at_most_one_constraints()
        self.check_food_group_constraints()
        self.check_objective_signs()
        
        # Generate detailed constraint comparison file
        self.generate_constraint_comparison_file()
        
        # Determine if validation passed
        is_valid = len(self.discrepancies) == 0
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        if is_valid:
            print(f"✅ PASSED - CQM matches PuLP formulation")
            if self.warnings:
                print(f"⚠️  {len(self.warnings)} warnings (non-critical)")
        else:
            print(f"❌ FAILED - Found {len(self.discrepancies)} discrepancies")
            print(f"\nCRITICAL ISSUES:")
            for i, disc in enumerate(self.discrepancies, 1):
                print(f"\n{i}. {disc['type']} ({disc['severity']})")
                print(f"   {disc['message']}")
                if 'details' in disc:
                    print(f"   Details: {disc['details']}")
        
        if self.warnings:
            print(f"\nWARNINGS:")
            for i, warn in enumerate(self.warnings, 1):
                print(f"{i}. {warn['message']}")
        
        print(f"{'='*80}\n")
        
        return is_valid, self.discrepancies, self.warnings
    
    def check_constraint_count(self):
        """Check if constraint counts match."""
        pulp_count = len([c for c in self.pulp.constraints.values()])
        cqm_count = len(self.cqm.constraints)
        
        # Allow some flexibility as CQM might have additional variable bound constraints
        if abs(pulp_count - cqm_count) > 10:
            self.discrepancies.append({
                'type': 'constraint_count_mismatch',
                'severity': 'warning',
                'message': f"Constraint count differs significantly: PuLP={pulp_count}, CQM={cqm_count}",
                'pulp_count': pulp_count,
                'cqm_count': cqm_count
            })
        elif pulp_count != cqm_count:
            self.warnings.append({
                'type': 'constraint_count_difference',
                'message': f"Minor constraint count difference: PuLP={pulp_count}, CQM={cqm_count}"
            })
    
    def check_variable_count(self):
        """Check if variable counts match."""
        # Get PuLP variables
        pulp_vars = self.pulp.variables()
        pulp_count = len(pulp_vars)
        cqm_count = len(self.cqm.variables)
        
        if pulp_count != cqm_count:
            self.discrepancies.append({
                'type': 'variable_count_mismatch',
                'severity': 'error',
                'message': f"Variable count mismatch: PuLP={pulp_count}, CQM={cqm_count}",
                'pulp_count': pulp_count,
                'cqm_count': cqm_count
            })
            print(f"  ❌ Variable count: PuLP={pulp_count}, CQM={cqm_count}")
        else:
            print(f"  ✓ Variable count matches: {pulp_count}")
    
    def check_at_most_one_constraints(self):
        """Check 'at most one' constraints for patch scenario."""
        if self.scenario_type != 'patch':
            return
        
        print(f"\n--- Checking 'At Most One Crop Per Plot' Constraints ---")
        
        # Find PuLP constraints for "at most one"
        pulp_at_most_one = {}
        for name, constraint in self.pulp.constraints.items():
            if 'Max_Assignment' in name or 'Max_Area' in name or 'MaxArea' in name:
                # Extract plot identifier from constraint name
                plot_id = self._extract_plot_id_from_name(name)
                pulp_at_most_one[plot_id] = (name, constraint)
        
        # Find CQM constraints for "at most one"
        cqm_at_most_one = {}
        for label, constraint in self.cqm.constraints.items():
            if ('AtMostOne' in label or '_One_' in label or 
                'MaxArea' in label or 'Max_Area' in label or 
                'Max_Assignment' in label or 'MaxAssignment' in label):
                # Extract plot identifier from constraint label
                plot_id = self._extract_plot_id_from_name(label)
                cqm_at_most_one[plot_id] = (label, constraint)
        
        print(f"  PuLP 'at most one' constraints: {len(pulp_at_most_one)}")
        print(f"  CQM 'at most one' constraints: {len(cqm_at_most_one)}")
        
        # Should have one per plot
        expected_count = self.info.get('n_units', 0)
        
        if len(cqm_at_most_one) != expected_count:
            self.discrepancies.append({
                'type': 'missing_at_most_one_constraints',
                'severity': 'critical',
                'message': f"Expected {expected_count} 'at most one' constraints, found {len(cqm_at_most_one)}",
                'expected': expected_count,
                'found': len(cqm_at_most_one)
            })
            print(f"  ❌ Expected {expected_count}, found {len(cqm_at_most_one)}")
            return
        
        # Compare matched constraints by plot ID
        print(f"  Comparing constraint expressions (LHS - RHS)...")
        issues_found = 0
        comparisons_done = 0
        
        # Get common plot IDs
        common_plots = set(pulp_at_most_one.keys()) & set(cqm_at_most_one.keys())
        
        # Sample a few to check (or all if small number)
        sample_size = min(5, len(common_plots))
        sample_plots = sorted(list(common_plots))[:sample_size]
        
        for plot_id in sample_plots:
            pulp_name, pulp_constraint = pulp_at_most_one[plot_id]
            cqm_label, cqm_constraint = cqm_at_most_one[plot_id]
            
            # Check constraint sense
            cqm_sense = cqm_constraint.sense.name
            if cqm_sense.upper() not in ['LE', 'LESS_EQUAL']:
                self.discrepancies.append({
                    'type': 'wrong_constraint_sense',
                    'severity': 'critical',
                    'message': f"Constraint {cqm_label} has wrong sense: {cqm_sense} (should be LE/LESS_EQUAL)",
                    'constraint': cqm_label,
                    'expected_sense': 'LE',
                    'actual_sense': cqm_sense
                })
                issues_found += 1
                print(f"  ❌ Plot {plot_id}: Wrong sense {cqm_sense}")
                continue
            
            # Extract and compare expressions (LHS - RHS)
            pulp_expr = self._extract_pulp_constraint_expression(pulp_constraint)
            cqm_expr = self._extract_cqm_constraint_expression(cqm_constraint)
            
            if not pulp_expr or not cqm_expr:
                self.warnings.append({
                    'type': 'constraint_extraction_failed',
                    'message': f"Could not extract expressions for plot {plot_id}"
                })
                continue
            
            # Compare expressions
            match, differences = self._compare_constraint_expressions(pulp_expr, cqm_expr)
            comparisons_done += 1
            
            if not match:
                self.discrepancies.append({
                    'type': 'constraint_expression_mismatch',
                    'severity': 'critical',
                    'message': f"Plot {plot_id}: Expressions differ between PuLP and CQM",
                    'constraint_pulp': pulp_name,
                    'constraint_cqm': cqm_label,
                    'differences': differences
                })
                issues_found += 1
                print(f"  ❌ Plot {plot_id}: Expression mismatch")
                for diff in differences[:3]:  # Show first 3 differences
                    print(f"      {diff}")
                if len(differences) > 3:
                    print(f"      ... and {len(differences) - 3} more differences")
        
        if issues_found == 0 and comparisons_done > 0:
            print(f"  ✓ All 'at most one' constraints match (checked {comparisons_done} samples)")
        elif comparisons_done == 0:
            self.warnings.append({
                'type': 'no_constraint_comparisons',
                'message': "Could not compare any 'at most one' constraints"
            })
            print(f"  ⚠️  Warning: Could not compare constraint expressions")
    
    def check_food_group_constraints(self):
        """Check food group diversity constraints."""
        print(f"\n--- Checking Food Group Constraints ---")
        
        # Find PuLP food group constraints - use constraint name as key to avoid overwriting
        pulp_fg = {}
        for name, constraint in self.pulp.constraints.items():
            if ('FoodGroup' in name or 'MinFoodGroup' in name or 'MaxFoodGroup' in name or 
                'Food_Group' in name or 'food_group' in name):
                # Use full constraint name as key to preserve both min and max
                pulp_fg[name] = (name, constraint)
        
        # Find CQM food group constraints - use constraint label as key
        cqm_fg = {}
        for label, constraint in self.cqm.constraints.items():
            if ('FoodGroup' in label or 'MinFoodGroup' in label or 'MaxFoodGroup' in label or
                'Food_Group' in label or 'Food Group' in label or 'food_group' in label):
                # Use full constraint label as key to preserve both min and max
                cqm_fg[label] = (label, constraint)
        
        print(f"  PuLP food group constraints: {len(pulp_fg)}")
        print(f"  CQM food group constraints: {len(cqm_fg)}")
        
        # Show samples for debugging
        if pulp_fg:
            sample_names = list(pulp_fg.values())[:3]
            print(f"  PuLP samples: {[name for name, _ in sample_names]}")
        if cqm_fg:
            sample_labels = list(cqm_fg.values())[:3]
            print(f"  CQM samples: {[label for label, _ in sample_labels]}")
        
        if len(pulp_fg) != len(cqm_fg):
            self.discrepancies.append({
                'type': 'food_group_count_mismatch',
                'severity': 'error',
                'message': f"Food group constraint count mismatch: PuLP={len(pulp_fg)}, CQM={len(cqm_fg)}",
                'pulp_count': len(pulp_fg),
                'cqm_count': len(cqm_fg)
            })
            print(f"  ❌ Count mismatch")
        else:
            print(f"  ✓ Count matches: {len(cqm_fg)}")
        
        # Compare expressions for matching food groups
        if pulp_fg and cqm_fg:
            print(f"  Comparing food group constraint expressions...")
            common_fgs = set(pulp_fg.keys()) & set(cqm_fg.keys())
            sample_size = min(3, len(common_fgs))
            sample_fgs = sorted(list(common_fgs))[:sample_size]
            
            issues_found = 0
            comparisons_done = 0
            
            for fg_id in sample_fgs:
                pulp_name, pulp_constraint = pulp_fg[fg_id]
                cqm_label, cqm_constraint = cqm_fg[fg_id]
                
                # Extract and compare expressions
                pulp_expr = self._extract_pulp_constraint_expression(pulp_constraint)
                cqm_expr = self._extract_cqm_constraint_expression(cqm_constraint)
                
                if not pulp_expr or not cqm_expr:
                    continue
                
                match, differences = self._compare_constraint_expressions(pulp_expr, cqm_expr)
                comparisons_done += 1
                
                if not match:
                    self.discrepancies.append({
                        'type': 'food_group_expression_mismatch',
                        'severity': 'error',
                        'message': f"Food group {fg_id}: Expressions differ between PuLP and CQM",
                        'constraint_pulp': pulp_name,
                        'constraint_cqm': cqm_label,
                        'differences': differences[:5]  # Limit to first 5 differences
                    })
                    issues_found += 1
                    print(f"  ❌ Food group {fg_id}: Expression mismatch")
            
            if issues_found == 0 and comparisons_done > 0:
                print(f"  ✓ Food group constraint expressions match (checked {comparisons_done} samples)")
    
    def _extract_food_group_id_from_name(self, name: str) -> str:
        """
        Extract food group identifier from constraint name.
        
        Examples:
            "MinFoodGroup_Vegetables" -> "Vegetables"
            "Food_Group_Grains_Min" -> "Grains"
        """
        import re
        # Remove common prefixes and suffixes
        clean_name = name.replace('MinFoodGroup_', '').replace('MaxFoodGroup_', '')
        clean_name = clean_name.replace('FoodGroup_', '').replace('Food_Group_', '')
        clean_name = clean_name.replace('_Min', '').replace('_Max', '')
        
        # Try to extract meaningful identifier (not just numbers)
        # If there's a recognizable food group name, use it
        parts = re.split(r'[_\s]+', clean_name)
        for part in parts:
            if part and not part.isdigit():
                return part
        
        # Fallback to full cleaned name
        return clean_name if clean_name else name
    
    def check_objective_signs(self):
        """Check if objective is being maximized with correct signs."""
        print(f"\n--- Checking Objective Function ---")
        
        # PuLP should be maximizing
        pulp_sense = self.pulp.sense
        pulp_is_max = (pulp_sense == 1 or pulp_sense == -1)  # LpMaximize = -1 in PuLP
        
        # CQM minimizes negative objective (equivalent to maximization)
        # Check objective coefficients
        cqm_obj = self.cqm.objective
        
        # Sample a few linear coefficients
        if hasattr(cqm_obj, 'linear'):
            sample_coeffs = list(cqm_obj.linear.values())[:5]
            if sample_coeffs:
                avg_coeff = sum(sample_coeffs) / len(sample_coeffs)
                
                # For maximization in CQM, we negate, so coefficients should be negative
                if avg_coeff > 0:
                    self.warnings.append({
                        'type': 'objective_sign_warning',
                        'message': f"CQM objective coefficients are positive (avg={avg_coeff:.4f}). Verify objective is correctly negated for maximization."
                    })
                    print(f"  ⚠️  CQM objective coefficients positive (might need negation)")
                else:
                    print(f"  ✓ CQM objective appears correctly negated for maximization")
        
        print(f"  PuLP sense: {'Maximize' if pulp_is_max else 'Minimize'}")
    
    def generate_constraint_comparison_file(self):
        """Generate detailed constraint comparison file."""
        print(f"\n--- Generating Constraint Comparison File ---")
        
        # Determine output directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(project_root, 'Benchmarks', 'COMPREHENSIVE', 'constraint_comparisons')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_units = self.info.get('n_units', 'unknown')
        filename = f"constraint_comparison_{self.scenario_type}_{n_units}units_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Build comparison report
        report_lines = []
        report_lines.append("="*100)
        report_lines.append(f"CONSTRAINT COMPARISON: PuLP vs CQM - {self.scenario_type.upper()} SCENARIO")
        report_lines.append("="*100)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Configuration: {n_units} units")
        report_lines.append(f"Total foods: {self.info.get('n_foods', 'unknown')}")
        report_lines.append("")
        
        # Extract PuLP constraints
        pulp_constraints_list = []
        for name, constraint in self.pulp.constraints.items():
            try:
                # Get constraint representation
                const_str = str(constraint)
                pulp_constraints_list.append({
                    'name': name,
                    'representation': const_str,
                    'type': self._categorize_constraint_name(name)
                })
            except Exception as e:
                pulp_constraints_list.append({
                    'name': name,
                    'representation': f"<Error extracting constraint: {e}>",
                    'type': 'error'
                })
        
        # Extract CQM constraints
        cqm_constraints_list = []
        for label, constraint in self.cqm.constraints.items():
            try:
                sense = constraint.sense.name
                rhs = constraint.rhs
                lhs = constraint.lhs
                num_vars = len(lhs.variables) if hasattr(lhs, 'variables') else 0
                
                cqm_constraints_list.append({
                    'name': label,
                    'sense': sense,
                    'rhs': rhs,
                    'num_variables': num_vars,
                    'type': self._categorize_constraint_name(label)
                })
            except Exception as e:
                cqm_constraints_list.append({
                    'name': label,
                    'representation': f"<Error extracting constraint: {e}>",
                    'type': 'error'
                })
        
        # SECTION 1: PuLP Constraints
        report_lines.append("\n" + "="*100)
        report_lines.append(f"SECTION 1: PuLP CONSTRAINTS (Total: {len(pulp_constraints_list)})")
        report_lines.append("="*100)
        report_lines.append("")
        
        # Group PuLP constraints by type
        pulp_by_type = defaultdict(list)
        for c in pulp_constraints_list:
            pulp_by_type[c['type']].append(c)
        
        for ctype in sorted(pulp_by_type.keys()):
            constraints = pulp_by_type[ctype]
            report_lines.append(f"\n--- {ctype.upper()} ({len(constraints)} constraints) ---")
            for i, c in enumerate(constraints, 1):
                report_lines.append(f"\n{i}. {c['name']}")
                report_lines.append(f"   {c['representation'][:200]}")  # Limit length
                if len(c['representation']) > 200:
                    report_lines.append(f"   ... (truncated, total length: {len(c['representation'])})")
        
        # SECTION 2: CQM Constraints
        report_lines.append("\n\n" + "="*100)
        report_lines.append(f"SECTION 2: CQM CONSTRAINTS (Total: {len(cqm_constraints_list)})")
        report_lines.append("="*100)
        report_lines.append("")
        
        # Group CQM constraints by type
        cqm_by_type = defaultdict(list)
        for c in cqm_constraints_list:
            cqm_by_type[c['type']].append(c)
        
        for ctype in sorted(cqm_by_type.keys()):
            constraints = cqm_by_type[ctype]
            report_lines.append(f"\n--- {ctype.upper()} ({len(constraints)} constraints) ---")
            for i, c in enumerate(constraints, 1):
                report_lines.append(f"\n{i}. {c['name']}")
                report_lines.append(f"   Sense: {c.get('sense', 'N/A')}")
                report_lines.append(f"   RHS: {c.get('rhs', 'N/A')}")
                report_lines.append(f"   Variables: {c.get('num_variables', 'N/A')}")
        
        # SECTION 3: Comparison Summary
        report_lines.append("\n\n" + "="*100)
        report_lines.append("SECTION 3: COMPARISON SUMMARY")
        report_lines.append("="*100)
        report_lines.append("")
        
        # Compare constraint types
        pulp_types = set(pulp_by_type.keys())
        cqm_types = set(cqm_by_type.keys())
        
        report_lines.append("Constraint Type Comparison:")
        report_lines.append(f"  PuLP types: {sorted(pulp_types)}")
        report_lines.append(f"  CQM types: {sorted(cqm_types)}")
        report_lines.append("")
        
        # Types only in PuLP
        only_in_pulp = pulp_types - cqm_types
        if only_in_pulp:
            report_lines.append(f"  ❌ Only in PuLP: {sorted(only_in_pulp)}")
            for ctype in sorted(only_in_pulp):
                report_lines.append(f"     - {ctype}: {len(pulp_by_type[ctype])} constraints")
        
        # Types only in CQM
        only_in_cqm = cqm_types - pulp_types
        if only_in_cqm:
            report_lines.append(f"  ⚠️  Only in CQM: {sorted(only_in_cqm)}")
            for ctype in sorted(only_in_cqm):
                report_lines.append(f"     - {ctype}: {len(cqm_by_type[ctype])} constraints")
        
        # Types in both
        in_both = pulp_types & cqm_types
        if in_both:
            report_lines.append(f"\n  ✓ In both: {sorted(in_both)}")
            for ctype in sorted(in_both):
                pulp_count = len(pulp_by_type[ctype])
                cqm_count = len(cqm_by_type[ctype])
                match_symbol = "✓" if pulp_count == cqm_count else "❌"
                report_lines.append(f"     {match_symbol} {ctype}: PuLP={pulp_count}, CQM={cqm_count}")
        
        # SECTION 3.5: Detailed Expression Comparison
        report_lines.append("\n\n" + "="*100)
        report_lines.append("SECTION 3.5: DETAILED EXPRESSION COMPARISON (LHS - RHS)")
        report_lines.append("="*100)
        report_lines.append("")
        report_lines.append("Comparing normalized expressions (LHS - RHS) to verify mathematical equivalence")
        report_lines.append("regardless of term ordering or constant placement.")
        report_lines.append("")
        
        # Compare at-most-one constraints by plot
        if self.scenario_type == 'patch':
            report_lines.append("\n--- At Most One Constraints (by Plot) ---")
            
            pulp_amo = {}
            for name, constraint in self.pulp.constraints.items():
                if any(keyword in name for keyword in ['Max_Assignment', 'Max_Area', 'MaxArea']):
                    plot_id = self._extract_plot_id_from_name(name)
                    pulp_amo[plot_id] = (name, constraint)
            
            cqm_amo = {}
            for label, constraint in self.cqm.constraints.items():
                if any(keyword in label for keyword in ['AtMostOne', '_One_', 'MaxArea', 'Max_Area', 'Max_Assignment', 'MaxAssignment']):
                    plot_id = self._extract_plot_id_from_name(label)
                    cqm_amo[plot_id] = (label, constraint)
            
            common_plots = sorted(set(pulp_amo.keys()) & set(cqm_amo.keys()))
            sample_plots = common_plots[:10]  # Compare up to 10 plots
            
            for plot_id in sample_plots:
                pulp_name, pulp_const = pulp_amo[plot_id]
                cqm_label, cqm_const = cqm_amo[plot_id]
                
                report_lines.append(f"\nPlot {plot_id}:")
                report_lines.append(f"  PuLP: {pulp_name}")
                report_lines.append(f"  CQM:  {cqm_label}")
                
                # Extract expressions
                pulp_expr = self._extract_pulp_constraint_expression(pulp_const)
                cqm_expr = self._extract_cqm_constraint_expression(cqm_const)
                
                if pulp_expr and cqm_expr:
                    match, differences = self._compare_constraint_expressions(pulp_expr, cqm_expr)
                    
                    if match:
                        report_lines.append(f"  Status: ✓ MATCH")
                    else:
                        report_lines.append(f"  Status: ❌ MISMATCH")
                        report_lines.append(f"  Differences ({len(differences)}):")
                        for diff in differences[:5]:
                            report_lines.append(f"    - {diff}")
                        if len(differences) > 5:
                            report_lines.append(f"    ... and {len(differences) - 5} more")
                else:
                    report_lines.append(f"  Status: ⚠️  Could not extract expressions")
        
        # Compare food group constraints
        report_lines.append("\n--- Food Group Constraints ---")
        
        pulp_fg = {}
        for name, constraint in self.pulp.constraints.items():
            if any(keyword in name for keyword in ['FoodGroup', 'MinFoodGroup', 'MaxFoodGroup', 'Food_Group', 'food_group']):
                fg_id = self._extract_food_group_id_from_name(name)
                pulp_fg[fg_id] = (name, constraint)
        
        cqm_fg = {}
        for label, constraint in self.cqm.constraints.items():
            if any(keyword in label for keyword in ['FoodGroup', 'MinFoodGroup', 'MaxFoodGroup', 'Food_Group', 'Food Group', 'food_group']):
                fg_id = self._extract_food_group_id_from_name(label)
                cqm_fg[fg_id] = (label, constraint)
        
        common_fgs = sorted(set(pulp_fg.keys()) & set(cqm_fg.keys()))
        sample_fgs = common_fgs[:5]  # Compare up to 5 food groups
        
        for fg_id in sample_fgs:
            pulp_name, pulp_const = pulp_fg[fg_id]
            cqm_label, cqm_const = cqm_fg[fg_id]
            
            report_lines.append(f"\nFood Group '{fg_id}':")
            report_lines.append(f"  PuLP: {pulp_name}")
            report_lines.append(f"  CQM:  {cqm_label}")
            
            # Extract expressions
            pulp_expr = self._extract_pulp_constraint_expression(pulp_const)
            cqm_expr = self._extract_cqm_constraint_expression(cqm_const)
            
            if pulp_expr and cqm_expr:
                match, differences = self._compare_constraint_expressions(pulp_expr, cqm_expr)
                
                if match:
                    report_lines.append(f"  Status: ✓ MATCH")
                else:
                    report_lines.append(f"  Status: ❌ MISMATCH")
                    report_lines.append(f"  Differences ({len(differences)}):")
                    for diff in differences[:5]:
                        report_lines.append(f"    - {diff}")
                    if len(differences) > 5:
                        report_lines.append(f"    ... and {len(differences) - 5} more")
            else:
                report_lines.append(f"  Status: ⚠️  Could not extract expressions")
        
        # SECTION 4: Discrepancies
        if self.discrepancies:
            report_lines.append("\n\n" + "="*100)
            report_lines.append(f"SECTION 4: VALIDATION DISCREPANCIES ({len(self.discrepancies)})")
            report_lines.append("="*100)
            report_lines.append("")
            
            for i, disc in enumerate(self.discrepancies, 1):
                report_lines.append(f"\n{i}. {disc['type'].upper()} [{disc['severity']}]")
                report_lines.append(f"   {disc['message']}")
                for key, value in disc.items():
                    if key not in ['type', 'severity', 'message']:
                        report_lines.append(f"   {key}: {value}")
        
        # Write to file with UTF-8 encoding to handle special characters
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✓ Saved detailed comparison to:")
        print(f"    {filepath}")
    
    def _categorize_constraint_name(self, name: str) -> str:
        """Categorize constraint by name pattern."""
        name_lower = name.lower()
        
        # Check for land availability constraints FIRST (before max_area check)
        if ('land_availability' in name_lower or 
            (name_lower.startswith('max_area_farm') and name_lower.count('_') == 2)):
            return 'land_availability'
        # Check for "at most one" constraints (various naming patterns)
        # Exclude max_area_if_selected which are coupling constraints
        elif (('max_assignment' in name_lower or 
               'maxassignment' in name_lower or 
               'one' in name_lower or 
               'atmostone' in name_lower) and
              'if_selected' not in name_lower):
            return 'at_most_one'
        # Check for coupling constraints (linking binary selection to continuous area)
        elif (('min_area_if_selected' in name_lower or
               'max_area_if_selected' in name_lower or
               'minarea_' in name_lower or
               'maxarea_' in name_lower) and
              name_lower.count('_') >= 3):  # Farm and food in name
            return 'coupling'
        # Check for food group constraints
        elif ('foodgroup' in name_lower or 
              'food_group' in name_lower or
              'food group' in name_lower):
            return 'food_group'
        # Check for minimum planting area constraints (plot-level, not coupling)
        elif ('min_plots' in name_lower or 
              'minplots' in name_lower):
            return 'min_planting_area'
        # Check for coupling/selection constraints
        elif 'selection' in name_lower:
            return 'coupling'
        else:
            return 'other'


def validate_before_dwave_submission(
    cqm,
    pulp_model,
    scenario_type: str,
    scenario_info: Dict,
    strict: bool = True
) -> bool:
    """
    Validate CQM against PuLP before D-Wave submission.
    
    Args:
        cqm: ConstrainedQuadraticModel
        pulp_model: PuLP model
        scenario_type: 'farm' or 'patch'
        scenario_info: Scenario metadata
        strict: If True, stop on any discrepancy. If False, only stop on critical issues.
        
    Returns:
        bool: True if validation passed, False otherwise
    """
    validator = CQMPuLPValidator(cqm, pulp_model, scenario_type, scenario_info)
    is_valid, discrepancies, warnings = validator.validate()
    
    if not is_valid:
        print(f"\n{'='*80}")
        print(f"⛔ STOPPING BENCHMARK - CQM VALIDATION FAILED")
        print(f"{'='*80}")
        print(f"\nThe CQM formulation does not match the PuLP formulation.")
        print(f"Submitting to D-Wave would waste solver time and budget.")
        print(f"\nPlease fix the discrepancies in solver_runner_BINARY.py")
        print(f"and re-run the benchmark.")
        print(f"\nDiscrepancies found: {len(discrepancies)}")
        for disc in discrepancies:
            if disc['severity'] in ['critical', 'error']:
                print(f"  • {disc['message']}")
        print(f"{'='*80}\n")
        
        if strict:
            return False
        else:
            # In non-strict mode, only stop on critical issues
            critical = [d for d in discrepancies if d['severity'] == 'critical']
            if critical:
                return False
    
    return True


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("CQM vs PuLP VALIDATOR")
    print("="*80)
    print()
    print("This utility validates CQM constraints against PuLP formulation")
    print("to prevent submitting incorrectly formulated problems to D-Wave.")
    print()
    print("Usage:")
    print("  from Utils.validate_cqm_vs_pulp import validate_before_dwave_submission")
    print()
    print("  # After creating CQM and PuLP models")
    print("  is_valid = validate_before_dwave_submission(")
    print("      cqm=cqm,")
    print("      pulp_model=pulp_model,")
    print("      scenario_type='patch',")
    print("      scenario_info={'n_units': 10, 'n_foods': 27}")
    print("  )")
    print()
    print("  if not is_valid:")
    print("      print('Validation failed - not submitting to D-Wave')")
    print("      sys.exit(1)")
    print()
    print("="*80)
