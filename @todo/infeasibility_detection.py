"""
Infeasibility Detection and Diagnosis Module

Provides tools for:
- Detecting infeasible optimization problems
- Diagnosing which constraints are conflicting
- Suggesting constraint relaxations
- Computing Irreducible Inconsistent Subsystems (IIS)
"""
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
import json


class InfeasibilityDiagnostic:
    """Diagnostic information for infeasible problems."""
    
    def __init__(self):
        self.is_infeasible = False
        self.conflicting_constraints = []
        self.iis_constraints = []
        self.relaxation_suggestions = []
        self.diagnostic_report = {}
    
    def to_dict(self) -> Dict:
        """Convert diagnostic to dictionary format."""
        return {
            'is_infeasible': self.is_infeasible,
            'num_conflicts': len(self.conflicting_constraints),
            'conflicting_constraints': self.conflicting_constraints,
            'iis_constraints': self.iis_constraints,
            'relaxation_suggestions': self.relaxation_suggestions,
            'diagnostic_report': self.diagnostic_report
        }
    
    def save_json(self, filepath: str):
        """Save diagnostic report to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


def detect_infeasibility(
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    compute_iis: bool = True
) -> InfeasibilityDiagnostic:
    """
    Detect and diagnose infeasibility in the farm allocation problem.
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names
        food_groups: Dictionary of food group constraints
        config: Configuration dictionary
        compute_iis: Whether to compute Irreducible Inconsistent Subsystem
    
    Returns:
        InfeasibilityDiagnostic object with detailed analysis
    """
    diagnostic = InfeasibilityDiagnostic()
    
    # Extract parameters
    params = config.get('parameters', {})
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    benefits = config.get('benefits', {})
    
    # Create model
    model = gp.Model("Infeasibility_Check")
    model.setParam('OutputFlag', 0)
    
    # Variables
    A = {}
    Y = {}
    for farm in farms:
        for food in foods:
            A[(farm, food)] = model.addVar(lb=0.0, name=f"A_{farm}_{food}")
            Y[(farm, food)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{farm}_{food}")
    
    # Objective (feasibility check - any objective works)
    model.setObjective(
        gp.quicksum(A[(f, c)] * benefits.get(c, 1.0) for f in farms for c in foods),
        GRB.MAXIMIZE
    )
    
    # Add constraints and track them
    constraint_map = {}
    
    # Track constraint objects for later IIS access
    constraint_objects = []
    
    # 1. Land availability
    for farm, capacity in farms.items():
        constr = model.addConstr(
            gp.quicksum(A[(farm, food)] for food in foods) <= capacity,
            name=f"Land_{farm}"
        )
        constraint_objects.append(constr)
        constraint_map[f"Land_{farm}"] = {
            'type': 'land_availability',
            'farm': farm,
            'capacity': capacity,
            'description': f"Total allocation on {farm} must not exceed {capacity:.2f}"
        }
    
    # 2. Min area if selected
    for farm in farms:
        for food in foods:
            min_area = min_planting_area.get(food, 0.0001)
            constr = model.addConstr(
                A[(farm, food)] >= min_area * Y[(farm, food)],
                name=f"MinArea_{farm}_{food}"
            )
            constraint_objects.append(constr)
            constraint_map[f"MinArea_{farm}_{food}"] = {
                'type': 'min_area',
                'farm': farm,
                'food': food,
                'min_value': min_area,
                'description': f"If {food} selected on {farm}, must plant at least {min_area:.4f}"
            }
    
    # 3. Max area if selected
    for farm in farms:
        for food in foods:
            max_area = max_planting_area.get(food, farms[farm])
            constr = model.addConstr(
                A[(farm, food)] <= max_area * Y[(farm, food)],
                name=f"MaxArea_{farm}_{food}"
            )
            constraint_objects.append(constr)
            constraint_map[f"MaxArea_{farm}_{food}"] = {
                'type': 'max_area',
                'farm': farm,
                'food': food,
                'max_value': max_area,
                'description': f"If {food} selected on {farm}, cannot exceed {max_area:.4f}"
            }
    
    # 4. Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            # Min constraint
            if min_foods > 0:
                constr = model.addConstr(
                    gp.quicksum(Y[(f, c)] for f in farms for c in foods_in_group) >= min_foods,
                    name=f"FG_Min_{group_name}"
                )
                constraint_objects.append(constr)
                constraint_map[f"FG_Min_{group_name}"] = {
                    'type': 'food_group_min',
                    'group': group_name,
                    'min_value': min_foods,
                    'foods': foods_in_group,
                    'description': f"Must select at least {min_foods} from {group_name}"
                }
            
            # Max constraint
            if max_foods < float('inf'):
                constr = model.addConstr(
                    gp.quicksum(Y[(f, c)] for f in farms for c in foods_in_group) <= max_foods,
                    name=f"FG_Max_{group_name}"
                )
                constraint_objects.append(constr)
                constraint_map[f"FG_Max_{group_name}"] = {
                    'type': 'food_group_max',
                    'group': group_name,
                    'max_value': max_foods,
                    'foods': foods_in_group,
                    'description': f"Cannot select more than {max_foods} from {group_name}"
                }
    
    # Solve model
    model.optimize()
    
    # Check for infeasibility
    if model.status == GRB.INFEASIBLE:
        diagnostic.is_infeasible = True
        print("\n" + "="*80)
        print("INFEASIBILITY DETECTED")
        print("="*80)
        
        if compute_iis:
            # Compute IIS
            print("\nComputing Irreducible Inconsistent Subsystem (IIS)...")
            model.computeIIS()
            
            # Extract IIS constraints
            all_constraints = model.getConstrs()
            for i, constr in enumerate(all_constraints):
                if constr.IISConstr:
                    # Get constraint name from index
                    constr_name = constr.ConstrName
                    constr_info = constraint_map.get(constr_name, {})
                    
                    diagnostic.iis_constraints.append({
                        'name': constr_name,
                        **constr_info
                    })
                    diagnostic.conflicting_constraints.append(constr_name)
            
            print(f"\nFound {len(diagnostic.iis_constraints)} constraints in IIS:")
            for constr_info in diagnostic.iis_constraints:
                print(f"  • {constr_info['name']}: {constr_info.get('description', 'N/A')}")
        
        # Generate relaxation suggestions
        diagnostic.relaxation_suggestions = generate_relaxation_suggestions(
            diagnostic.iis_constraints, farms, foods, config
        )
        
        print("\nSuggested Relaxations:")
        for i, suggestion in enumerate(diagnostic.relaxation_suggestions, 1):
            print(f"\n{i}. {suggestion['title']}")
            print(f"   {suggestion['description']}")
            if 'action' in suggestion:
                print(f"   Action: {suggestion['action']}")
    
    elif model.status == GRB.OPTIMAL:
        diagnostic.is_infeasible = False
        print("\nProblem is FEASIBLE")
        print(f"Optimal objective: {model.ObjVal:.4f}")
    
    else:
        diagnostic.diagnostic_report['solver_status'] = model.status
        diagnostic.diagnostic_report['solver_status_name'] = get_status_name(model.status)
        print(f"\nUnexpected solver status: {get_status_name(model.status)}")
    
    # Additional diagnostics
    diagnostic.diagnostic_report['total_constraints'] = model.NumConstrs
    diagnostic.diagnostic_report['total_variables'] = model.NumVars
    diagnostic.diagnostic_report['num_farms'] = len(farms)
    diagnostic.diagnostic_report['num_foods'] = len(foods)
    diagnostic.diagnostic_report['total_land'] = sum(farms.values())
    
    return diagnostic


def generate_relaxation_suggestions(
    iis_constraints: List[Dict],
    farms: Dict,
    foods: List[str],
    config: Dict
) -> List[Dict]:
    """
    Generate suggestions for relaxing constraints to make problem feasible.
    """
    suggestions = []
    
    # Analyze IIS constraints by type
    constraint_types = {}
    for constr in iis_constraints:
        constr_type = constr.get('type', 'unknown')
        if constr_type not in constraint_types:
            constraint_types[constr_type] = []
        constraint_types[constr_type].append(constr)
    
    # Suggestion 1: Relax land availability
    if 'land_availability' in constraint_types:
        farms_affected = [c['farm'] for c in constraint_types['land_availability']]
        suggestions.append({
            'title': 'Increase Land Availability',
            'type': 'land_relaxation',
            'description': f"Increase capacity for farms: {', '.join(farms_affected)}",
            'action': 'Add more land or reduce allocation requirements',
            'affected_farms': farms_affected
        })
    
    # Suggestion 2: Relax food group minimums
    if 'food_group_min' in constraint_types:
        groups_affected = [c['group'] for c in constraint_types['food_group_min']]
        suggestions.append({
            'title': 'Reduce Food Group Minimums',
            'type': 'food_group_min_relaxation',
            'description': f"Reduce minimum requirements for: {', '.join(groups_affected)}",
            'action': 'Lower min_foods parameter in config',
            'affected_groups': groups_affected
        })
    
    # Suggestion 3: Increase food group maximums
    if 'food_group_max' in constraint_types:
        groups_affected = [c['group'] for c in constraint_types['food_group_max']]
        suggestions.append({
            'title': 'Increase Food Group Maximums',
            'type': 'food_group_max_relaxation',
            'description': f"Increase maximum limits for: {', '.join(groups_affected)}",
            'action': 'Raise max_foods parameter in config',
            'affected_groups': groups_affected
        })
    
    # Suggestion 4: Adjust min/max planting areas
    if 'min_area' in constraint_types or 'max_area' in constraint_types:
        suggestions.append({
            'title': 'Adjust Min/Max Planting Areas',
            'type': 'area_bounds_relaxation',
            'description': 'Some foods have incompatible min/max area requirements',
            'action': 'Review minimum_planting_area and maximum_planting_area in config'
        })
    
    # Suggestion 5: General relaxation
    if not suggestions:
        suggestions.append({
            'title': 'General Constraint Relaxation',
            'type': 'general',
            'description': 'Problem constraints are too tight',
            'action': 'Consider relaxing multiple constraints or simplifying the problem'
        })
    
    return suggestions


def get_status_name(status_code: int) -> str:
    """Convert Gurobi status code to readable name."""
    status_map = {
        GRB.OPTIMAL: 'OPTIMAL',
        GRB.INFEASIBLE: 'INFEASIBLE',
        GRB.INF_OR_UNBD: 'INFEASIBLE_OR_UNBOUNDED',
        GRB.UNBOUNDED: 'UNBOUNDED',
        GRB.CUTOFF: 'CUTOFF',
        GRB.ITERATION_LIMIT: 'ITERATION_LIMIT',
        GRB.NODE_LIMIT: 'NODE_LIMIT',
        GRB.TIME_LIMIT: 'TIME_LIMIT',
        GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT',
        GRB.INTERRUPTED: 'INTERRUPTED',
        GRB.NUMERIC: 'NUMERIC_ISSUES',
        GRB.SUBOPTIMAL: 'SUBOPTIMAL',
        GRB.INPROGRESS: 'IN_PROGRESS'
    }
    return status_map.get(status_code, f'UNKNOWN_STATUS_{status_code}')


def check_config_feasibility(config: Dict, farms: Dict, foods: List[str]) -> Dict:
    """
    Quick feasibility check on configuration parameters.
    
    Returns diagnostic dictionary with warnings.
    """
    warnings = []
    params = config.get('parameters', {})
    
    min_planting = params.get('minimum_planting_area', {})
    max_planting = params.get('maximum_planting_area', {})
    
    # Check 1: Min > Max for any food
    for food in foods:
        min_val = min_planting.get(food, 0.0)
        max_val = max_planting.get(food, float('inf'))
        if min_val > max_val:
            warnings.append({
                'type': 'min_exceeds_max',
                'food': food,
                'min': min_val,
                'max': max_val,
                'message': f"Min area ({min_val}) > Max area ({max_val}) for {food}"
            })
    
    # Check 2: Total min requirements exceed total land
    total_land = sum(farms.values())
    for food in foods:
        min_val = min_planting.get(food, 0.0)
        if min_val * len(farms) > total_land:
            warnings.append({
                'type': 'insufficient_land',
                'food': food,
                'min_required': min_val * len(farms),
                'available': total_land,
                'message': f"Min requirements for {food} exceed total available land"
            })
    
    return {
        'is_feasible': len(warnings) == 0,
        'num_warnings': len(warnings),
        'warnings': warnings
    }
