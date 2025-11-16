"""
Constraint Validator for BQM_PATCH Solutions
Verifies that solutions respect all constraints defined in the problem.
"""

def validate_pulp_patch_constraints(X_variables, Y_variables, patches, foods, food_groups, config):
    """
    Validate all constraints for a PuLP BQM_PATCH solution.
    
    Args:
        X_variables: Dictionary of X variable values (key: (plot, crop) or "plot_crop")
        Y_variables: Dictionary of Y variable values (key: crop)
        patches: List of patch names
        foods: Dictionary of food data
        food_groups: Dictionary mapping groups to food lists
        config: Configuration dictionary
    
    Returns:
        Dictionary with validation results for each constraint type
    """
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    min_planting_area = params.get('minimum_planting_area', {})
    max_percentage_per_crop = params.get('max_percentage_per_crop', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    total_land = sum(land_availability.values())
    
    # Extract X and Y variables - handle both tuple keys and string keys
    X_vals = {}
    for key, value in X_variables.items():
        if isinstance(key, tuple):
            X_vals[key] = value
        else:
            # Parse string key like "plot_1_Wheat"
            parts = str(key).split('_')
            if len(parts) >= 2:
                # Reconstruct plot and crop names
                # Find where crop name starts (after plot identifier)
                plot = '_'.join(parts[:-1]) if parts[-1] in foods else parts[0]
                crop = parts[-1] if parts[-1] in foods else '_'.join(parts[1:])
                X_vals[(plot, crop)] = value
    
    Y_vals = {}
    for crop in foods:
        Y_vals[crop] = Y_variables.get(crop, 0)
    
    # Validation results
    validation = {
        'all_satisfied': True,
        'violations': [],
        'constraint_details': {}
    }
    
    # (1) At most one crop per plot: sum_c X_{p,c} <= 1
    at_most_one_violations = []
    for plot in patches:
        sum_x = sum(X_vals.get((plot, c), 0) for c in foods)
        satisfied = sum_x <= 1.01  # Small tolerance for numerical errors
        
        if not satisfied:
            at_most_one_violations.append({
                'plot': plot,
                'sum': sum_x,
                'limit': 1.0,
                'violation': sum_x - 1.0
            })
            validation['all_satisfied'] = False
    
    validation['constraint_details']['at_most_one_per_plot'] = {
        'satisfied': len(at_most_one_violations) == 0,
        'num_constraints': len(patches),
        'num_violations': len(at_most_one_violations),
        'violations': at_most_one_violations
    }
    
    # (2) X-Y Linking: X_{p,c} <= Y_c
    xy_linking_violations = []
    for plot in patches:
        for crop in foods:
            x_val = X_vals.get((plot, crop), 0)
            y_val = Y_vals[crop]
            satisfied = x_val <= y_val + 0.01  # Tolerance
            
            if not satisfied:
                xy_linking_violations.append({
                    'plot': plot,
                    'crop': crop,
                    'X_pc': x_val,
                    'Y_c': y_val,
                    'violation': x_val - y_val
                })
                validation['all_satisfied'] = False
    
    validation['constraint_details']['x_y_linking'] = {
        'satisfied': len(xy_linking_violations) == 0,
        'num_constraints': len(patches) * len(foods),
        'num_violations': len(xy_linking_violations),
        'violations': xy_linking_violations[:10]  # Limit to first 10
    }
    
    # (3) Y Activation: Y_c <= sum_p X_{p,c}
    y_activation_violations = []
    for crop in foods:
        sum_x = sum(X_vals.get((p, crop), 0) for p in patches)
        y_val = Y_vals[crop]
        satisfied = y_val <= sum_x + 0.01  # Tolerance
        
        if not satisfied:
            y_activation_violations.append({
                'crop': crop,
                'Y_c': y_val,
                'sum_X_pc': sum_x,
                'violation': y_val - sum_x
            })
            validation['all_satisfied'] = False
    
    validation['constraint_details']['y_activation'] = {
        'satisfied': len(y_activation_violations) == 0,
        'num_constraints': len(foods),
        'num_violations': len(y_activation_violations),
        'violations': y_activation_violations
    }
    
    # (4) Area bounds per crop
    area_bounds_violations = []
    crop_areas = {}
    
    for crop in foods:
        area = sum(land_availability[p] * X_vals.get((p, crop), 0) for p in patches)
        crop_areas[crop] = area
        
        # Get bounds
        min_area = min_planting_area.get(crop, 0.0)
        
        # Calculate max area
        if crop in max_percentage_per_crop:
            max_area = total_land * max_percentage_per_crop[crop]
        else:
            max_area = total_land
        
        # Check if Y_c = 1 (crop is selected)
        if Y_vals[crop] > 0.5:
            # Crop is selected, check bounds
            min_satisfied = area >= min_area - 0.01
            max_satisfied = area <= max_area + 0.01
            
            if not min_satisfied:
                area_bounds_violations.append({
                    'crop': crop,
                    'area': area,
                    'min_required': min_area,
                    'max_allowed': max_area,
                    'violation_type': 'min',
                    'violation': min_area - area
                })
                validation['all_satisfied'] = False
            
            if not max_satisfied:
                area_bounds_violations.append({
                    'crop': crop,
                    'area': area,
                    'min_required': min_area,
                    'max_allowed': max_area,
                    'violation_type': 'max',
                    'violation': area - max_area
                })
                validation['all_satisfied'] = False
    
    validation['constraint_details']['area_bounds'] = {
        'satisfied': len(area_bounds_violations) == 0,
        'num_constraints': len(foods) * 2,  # min and max for each
        'num_violations': len(area_bounds_violations),
        'violations': area_bounds_violations,
        'crop_areas': crop_areas
    }
    
    # (5) Food group diversity: FG_g^min <= sum_{c in G_g} Y_c <= FG_g^max
    food_group_violations = []
    
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            crops_in_group = food_groups.get(group, [])
            num_crops_selected = sum(Y_vals.get(c, 0) for c in crops_in_group if c in Y_vals)
            
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(crops_in_group))
            
            min_satisfied = num_crops_selected >= min_foods - 0.01
            max_satisfied = num_crops_selected <= max_foods + 0.01
            
            if not min_satisfied:
                food_group_violations.append({
                    'group': group,
                    'num_selected': num_crops_selected,
                    'min_required': min_foods,
                    'max_allowed': max_foods,
                    'violation_type': 'min',
                    'violation': min_foods - num_crops_selected
                })
                validation['all_satisfied'] = False
            
            if not max_satisfied:
                food_group_violations.append({
                    'group': group,
                    'num_selected': num_crops_selected,
                    'min_required': min_foods,
                    'max_allowed': max_foods,
                    'violation_type': 'max',
                    'violation': num_crops_selected - max_foods
                })
                validation['all_satisfied'] = False
    
    validation['constraint_details']['food_group_diversity'] = {
        'satisfied': len(food_group_violations) == 0,
        'num_constraints': len(food_group_constraints) * 2 if food_group_constraints else 0,
        'num_violations': len(food_group_violations),
        'violations': food_group_violations
    }
    
    # Summary statistics
    validation['summary'] = {
        'total_constraints': (
            len(patches) +  # at most one per plot
            len(patches) * len(foods) +  # X-Y linking
            len(foods) +  # Y activation
            len(foods) * 2 +  # area bounds
            (len(food_group_constraints) * 2 if food_group_constraints else 0)  # food groups
        ),
        'total_violations': (
            len(at_most_one_violations) +
            len(xy_linking_violations) +
            len(y_activation_violations) +
            len(area_bounds_violations) +
            len(food_group_violations)
        ),
        'plots_assigned': sum(1 for p in patches if sum(X_vals.get((p, c), 0) for c in foods) > 0.5),
        'crops_selected': sum(1 for c in foods if Y_vals[c] > 0.5),
        'total_area_used': sum(land_availability[p] * X_vals.get((p, c), 0) for p in patches for c in foods),
        'total_area_available': total_land,
        'area_utilization': sum(land_availability[p] * X_vals.get((p, c), 0) for p in patches for c in foods) / total_land * 100
    }
    
    return validation


def validate_bqm_patch_constraints(sample, invert, patches, foods, food_groups, config):
    """
    Validate all constraints for a BQM_PATCH solution.
    
    Args:
        sample: BQM sample dictionary
        invert: Invert function from cqm_to_bqm
        patches: List of patch names
        foods: Dictionary of food data
        food_groups: Dictionary mapping groups to food lists
        config: Configuration dictionary
    
    Returns:
        Dictionary with validation results for each constraint type
    """
    # Convert BQM sample back to CQM variables
    cqm_sample = invert(sample)
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    min_planting_area = params.get('minimum_planting_area', {})
    max_percentage_per_crop = params.get('max_percentage_per_crop', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    total_land = sum(land_availability.values())
    
    # Extract X and Y variables from sample
    X_vals = {}
    Y_vals = {}
    
    for plot in patches:
        for crop in foods:
            var_name = f"X_{plot}_{crop}"
            X_vals[(plot, crop)] = cqm_sample.get(var_name, 0)
    
    for crop in foods:
        var_name = f"Y_{crop}"
        Y_vals[crop] = cqm_sample.get(var_name, 0)
    
    # Validation results
    validation = {
        'all_satisfied': True,
        'violations': [],
        'constraint_details': {}
    }
    
    # (1) At most one crop per plot: sum_c X_{p,c} <= 1
    at_most_one_violations = []
    for plot in patches:
        sum_x = sum(X_vals[(plot, c)] for c in foods)
        satisfied = sum_x <= 1.01  # Small tolerance for numerical errors
        
        if not satisfied:
            at_most_one_violations.append({
                'plot': plot,
                'sum': sum_x,
                'limit': 1.0,
                'violation': sum_x - 1.0
            })
            validation['all_satisfied'] = False
    
    validation['constraint_details']['at_most_one_per_plot'] = {
        'satisfied': len(at_most_one_violations) == 0,
        'num_constraints': len(patches),
        'num_violations': len(at_most_one_violations),
        'violations': at_most_one_violations
    }
    
    # (2) X-Y Linking: X_{p,c} <= Y_c
    xy_linking_violations = []
    for plot in patches:
        for crop in foods:
            x_val = X_vals[(plot, crop)]
            y_val = Y_vals[crop]
            satisfied = x_val <= y_val + 0.01  # Tolerance
            
            if not satisfied:
                xy_linking_violations.append({
                    'plot': plot,
                    'crop': crop,
                    'X_pc': x_val,
                    'Y_c': y_val,
                    'violation': x_val - y_val
                })
                validation['all_satisfied'] = False
    
    validation['constraint_details']['x_y_linking'] = {
        'satisfied': len(xy_linking_violations) == 0,
        'num_constraints': len(patches) * len(foods),
        'num_violations': len(xy_linking_violations),
        'violations': xy_linking_violations[:10]  # Limit to first 10
    }
    
    # (3) Y Activation: Y_c <= sum_p X_{p,c}
    y_activation_violations = []
    for crop in foods:
        sum_x = sum(X_vals[(p, crop)] for p in patches)
        y_val = Y_vals[crop]
        satisfied = y_val <= sum_x + 0.01  # Tolerance
        
        if not satisfied:
            y_activation_violations.append({
                'crop': crop,
                'Y_c': y_val,
                'sum_X_pc': sum_x,
                'violation': y_val - sum_x
            })
            validation['all_satisfied'] = False
    
    validation['constraint_details']['y_activation'] = {
        'satisfied': len(y_activation_violations) == 0,
        'num_constraints': len(foods),
        'num_violations': len(y_activation_violations),
        'violations': y_activation_violations
    }
    
    # (4) Area bounds per crop
    area_bounds_violations = []
    crop_areas = {}
    
    for crop in foods:
        area = sum(land_availability[p] * X_vals[(p, crop)] for p in patches)
        crop_areas[crop] = area
        
        # Get bounds
        min_area = min_planting_area.get(crop, 0.0)
        
        # Calculate max area
        if crop in max_percentage_per_crop:
            max_area = total_land * max_percentage_per_crop[crop]
        else:
            max_area = total_land
        
        # Check if Y_c = 1 (crop is selected)
        if Y_vals[crop] > 0.5:
            # Crop is selected, check bounds
            min_satisfied = area >= min_area - 0.01
            max_satisfied = area <= max_area + 0.01
            
            if not min_satisfied:
                area_bounds_violations.append({
                    'crop': crop,
                    'area': area,
                    'min_required': min_area,
                    'max_allowed': max_area,
                    'violation_type': 'min',
                    'violation': min_area - area
                })
                validation['all_satisfied'] = False
            
            if not max_satisfied:
                area_bounds_violations.append({
                    'crop': crop,
                    'area': area,
                    'min_required': min_area,
                    'max_allowed': max_area,
                    'violation_type': 'max',
                    'violation': area - max_area
                })
                validation['all_satisfied'] = False
    
    validation['constraint_details']['area_bounds'] = {
        'satisfied': len(area_bounds_violations) == 0,
        'num_constraints': len(foods) * 2,  # min and max for each
        'num_violations': len(area_bounds_violations),
        'violations': area_bounds_violations,
        'crop_areas': crop_areas
    }
    
    # (5) Food group diversity: FG_g^min <= sum_{c in G_g} Y_c <= FG_g^max
    food_group_violations = []
    
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            crops_in_group = food_groups.get(group, [])
            num_crops_selected = sum(Y_vals.get(c, 0) for c in crops_in_group if c in Y_vals)
            
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(crops_in_group))
            
            min_satisfied = num_crops_selected >= min_foods - 0.01
            max_satisfied = num_crops_selected <= max_foods + 0.01
            
            if not min_satisfied:
                food_group_violations.append({
                    'group': group,
                    'num_selected': num_crops_selected,
                    'min_required': min_foods,
                    'max_allowed': max_foods,
                    'violation_type': 'min',
                    'violation': min_foods - num_crops_selected
                })
                validation['all_satisfied'] = False
            
            if not max_satisfied:
                food_group_violations.append({
                    'group': group,
                    'num_selected': num_crops_selected,
                    'min_required': min_foods,
                    'max_allowed': max_foods,
                    'violation_type': 'max',
                    'violation': num_crops_selected - max_foods
                })
                validation['all_satisfied'] = False
    
    validation['constraint_details']['food_group_diversity'] = {
        'satisfied': len(food_group_violations) == 0,
        'num_constraints': len(food_group_constraints) * 2 if food_group_constraints else 0,
        'num_violations': len(food_group_violations),
        'violations': food_group_violations
    }
    
    # Summary statistics
    validation['summary'] = {
        'total_constraints': (
            len(patches) +  # at most one per plot
            len(patches) * len(foods) +  # X-Y linking
            len(foods) +  # Y activation
            len(foods) * 2 +  # area bounds
            (len(food_group_constraints) * 2 if food_group_constraints else 0)  # food groups
        ),
        'total_violations': (
            len(at_most_one_violations) +
            len(xy_linking_violations) +
            len(y_activation_violations) +
            len(area_bounds_violations) +
            len(food_group_violations)
        ),
        'plots_assigned': sum(1 for p in patches if sum(X_vals[(p, c)] for c in foods) > 0.5),
        'crops_selected': sum(1 for c in foods if Y_vals[c] > 0.5),
        'total_area_used': sum(land_availability[p] * X_vals[(p, c)] for p in patches for c in foods),
        'total_area_available': total_land,
        'area_utilization': sum(land_availability[p] * X_vals[(p, c)] for p in patches for c in foods) / total_land * 100
    }
    
    return validation


def print_validation_report(validation, verbose=True):
    """
    Print a formatted validation report.
    
    Args:
        validation: Dictionary returned by validate_bqm_patch_constraints
        verbose: If True, print detailed violations
    """
    print("\n" + "="*80)
    print("CONSTRAINT VALIDATION REPORT")
    print("="*80)
    
    if validation['all_satisfied']:
        print("✅ ALL CONSTRAINTS SATISFIED!")
    else:
        print("❌ CONSTRAINT VIOLATIONS DETECTED!")
    
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    summary = validation['summary']
    print(f"  Total Constraints:    {summary['total_constraints']}")
    print(f"  Total Violations:     {summary['total_violations']}")
    print(f"  Plots Assigned:       {summary['plots_assigned']}")
    print(f"  Crops Selected:       {summary['crops_selected']}")
    print(f"  Area Used:            {summary['total_area_used']:.3f} ha")
    print(f"  Area Available:       {summary['total_area_available']:.3f} ha")
    print(f"  Utilization:          {summary['area_utilization']:.1f}%")
    
    # Print constraint-by-constraint details
    print("\n" + "-"*80)
    print("CONSTRAINT DETAILS")
    print("-"*80)
    
    details = validation['constraint_details']
    
    # 1. At most one crop per plot
    c1 = details['at_most_one_per_plot']
    status = "✅" if c1['satisfied'] else "❌"
    print(f"\n{status} (1) At Most One Crop Per Plot")
    print(f"     Constraints: {c1['num_constraints']}, Violations: {c1['num_violations']}")
    if verbose and c1['violations']:
        for v in c1['violations'][:5]:
            print(f"       - Plot {v['plot']}: sum={v['sum']:.3f} > limit={v['limit']:.3f}")
    
    # 2. X-Y Linking
    c2 = details['x_y_linking']
    status = "✅" if c2['satisfied'] else "❌"
    print(f"\n{status} (2) X-Y Linking (X_pc <= Y_c)")
    print(f"     Constraints: {c2['num_constraints']}, Violations: {c2['num_violations']}")
    if verbose and c2['violations']:
        for v in c2['violations'][:5]:
            print(f"       - Plot {v['plot']}, Crop {v['crop']}: X={v['X_pc']:.3f} > Y={v['Y_c']:.3f}")
    
    # 3. Y Activation
    c3 = details['y_activation']
    status = "✅" if c3['satisfied'] else "❌"
    print(f"\n{status} (3) Y Activation (Y_c <= sum_p X_pc)")
    print(f"     Constraints: {c3['num_constraints']}, Violations: {c3['num_violations']}")
    if verbose and c3['violations']:
        for v in c3['violations'][:5]:
            print(f"       - Crop {v['crop']}: Y={v['Y_c']:.3f} > sum_X={v['sum_X_pc']:.3f}")
    
    # 4. Area Bounds
    c4 = details['area_bounds']
    status = "✅" if c4['satisfied'] else "❌"
    print(f"\n{status} (4) Area Bounds Per Crop")
    print(f"     Constraints: {c4['num_constraints']}, Violations: {c4['num_violations']}")
    if verbose and c4['violations']:
        for v in c4['violations'][:5]:
            print(f"       - Crop {v['crop']}: area={v['area']:.3f}, {v['violation_type']}={v['min_required'] if v['violation_type']=='min' else v['max_allowed']:.3f}")
    
    # Show crop areas
    if verbose:
        print("\n     Crop Areas:")
        for crop, area in sorted(c4['crop_areas'].items(), key=lambda x: -x[1])[:10]:
            print(f"       {crop:20s}: {area:6.3f} ha")
    
    # 5. Food Group Diversity
    c5 = details['food_group_diversity']
    status = "✅" if c5['satisfied'] else "❌"
    print(f"\n{status} (5) Food Group Diversity")
    print(f"     Constraints: {c5['num_constraints']}, Violations: {c5['num_violations']}")
    if verbose and c5['violations']:
        for v in c5['violations'][:5]:
            print(f"       - Group {v['group']}: selected={v['num_selected']:.1f}, {v['violation_type']}={v['min_required'] if v['violation_type']=='min' else v['max_allowed']:.1f}")
    
    print("\n" + "="*80)
    
    return validation['all_satisfied']
