#!/usr/bin/env python3
"""
ILP Hardness Diagnostic Tool

Runs four diagnostic checks on an Integer Linear Program (ILP) to estimate
how "hard" it is for a branch-and-cut solver (e.g., Gurobi).

Sections:
1. Integrality (root) gap
2. Fractionality & root solution structure  
3. Structural properties of the constraint matrix
4. Symmetry & repeated blocks

Usage:
    python ilp_hardness_diagnostic.py <input_path> [--out_dir <dir>] [--time_limit <seconds>]
"""

import os
import sys
import json
import csv
import hashlib
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# ============================================================================
# CONFIGURATION (adjustable thresholds)
# ============================================================================

CONFIG = {
    'fractionality_tolerance': 1e-8,      # Threshold for detecting fractional values
    'big_M_multiplier': 1e3,              # Multiplier for big-M detection (vs median)
    'big_M_absolute_threshold': 1e6,      # Absolute threshold for big-M detection
    'hash_tolerance': 1e-6,               # Tolerance for coefficient hashing
    'mip_time_limit': 300,                # Time limit for MIP solve (seconds)
    'top_k_fractional': 20,               # Number of top fractional vars to report
    'min_symmetric_group_size': 2,        # Minimum size for symmetric groups
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

class Logger:
    """Simple logger that writes to both console and file."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.logs = []
        
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.logs.append(log_line)
        
    def save(self):
        with open(self.log_path, 'w') as f:
            f.write('\n'.join(self.logs))

# ============================================================================
# SECTION 1: Integrality (Root) Gap
# ============================================================================

def section1_integrality_gap(model, logger: Logger, report: Dict) -> Tuple[Any, Optional[float]]:
    """
    Compute LP relaxation objective, integer objective, and integrality gap.
    
    Returns:
        Tuple of (relaxed_model, lp_obj) for reuse in Section 2
    """
    from gurobipy import GRB
    
    logger.log("=" * 60)
    logger.log("SECTION 1: Integrality (Root) Gap")
    logger.log("=" * 60)
    
    errors = []
    int_obj = None
    lp_obj = None
    gap = None
    rel = None
    
    try:
        # 1) Solve MIP to get integer incumbent
        logger.log("Solving MIP to obtain integer incumbent...")
        model.setParam('TimeLimit', CONFIG['mip_time_limit'])
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            int_obj = model.ObjVal
            logger.log(f"MIP optimal objective: {int_obj:.6f}")
        elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
            int_obj = model.ObjVal
            logger.log(f"MIP time limit reached. Best incumbent: {int_obj:.6f}")
        elif model.status == GRB.INFEASIBLE:
            logger.log("MIP is INFEASIBLE")
            errors.append("MIP is infeasible")
        elif model.status == GRB.UNBOUNDED:
            logger.log("MIP is UNBOUNDED")
            errors.append("MIP is unbounded")
        else:
            logger.log(f"MIP solve status: {model.status}, no incumbent found")
            errors.append(f"MIP status {model.status}, no incumbent")
        
        # 2) LP relaxation
        logger.log("Creating and solving LP relaxation...")
        rel = model.relax()
        rel.setParam('OutputFlag', 0)
        rel.optimize()
        
        if rel.status == GRB.OPTIMAL:
            lp_obj = rel.ObjVal
            logger.log(f"LP relaxation objective: {lp_obj:.6f}")
        elif rel.status == GRB.INFEASIBLE:
            logger.log("LP relaxation is INFEASIBLE")
            errors.append("LP relaxation is infeasible")
        elif rel.status == GRB.UNBOUNDED:
            logger.log("LP relaxation is UNBOUNDED")
            errors.append("LP relaxation is unbounded")
        else:
            logger.log(f"LP relaxation status: {rel.status}")
            errors.append(f"LP relaxation status {rel.status}")
        
        # 3) Compute integrality gap
        if int_obj is not None and lp_obj is not None:
            # For maximization: gap = (lp_obj - int_obj) / max(1.0, abs(int_obj))
            # For minimization: gap = (int_obj - lp_obj) / max(1.0, abs(int_obj))
            is_maximize = model.ModelSense == GRB.MAXIMIZE
            if is_maximize:
                gap = (lp_obj - int_obj) / max(1.0, abs(int_obj))
            else:
                gap = (int_obj - lp_obj) / max(1.0, abs(int_obj))
            logger.log(f"Integrality gap: {gap:.6f} ({gap*100:.2f}%)")
        
    except Exception as e:
        logger.log(f"ERROR in Section 1: {str(e)}")
        errors.append(str(e))
    
    # Update report
    report['lp_obj'] = lp_obj
    report['int_obj'] = int_obj
    report['integrality_gap'] = gap
    if errors:
        report.setdefault('errors', []).extend(errors)
    
    logger.log(f"Section 1 complete: lp_obj={lp_obj}, int_obj={int_obj}, gap={gap}")
    
    return rel, lp_obj

# ============================================================================
# SECTION 2: Fractionality & Root Solution Structure
# ============================================================================

def section2_fractionality(rel, logger: Logger, report: Dict, out_dir: str):
    """
    Count fractional variables at LP root and identify most involved vars/constraints.
    """
    from gurobipy import GRB
    
    logger.log("=" * 60)
    logger.log("SECTION 2: Fractionality & Root Solution Structure")
    logger.log("=" * 60)
    
    if rel is None:
        logger.log("No LP relaxation available, skipping Section 2")
        report['fractional_count'] = None
        report['fractional_ratio'] = None
        return
    
    errors = []
    frac_tol = CONFIG['fractionality_tolerance']
    
    try:
        vars_rel = rel.getVars()
        total_vars = len(vars_rel)
        
        rows = []
        fractional_vars = []
        fractional_count = 0
        
        for v in vars_rel:
            val = v.x
            rounded = round(val)
            is_frac = abs(val - rounded) > frac_tol
            
            # Map VType to string
            vtype_map = {GRB.BINARY: 'B', GRB.INTEGER: 'I', GRB.CONTINUOUS: 'C'}
            vtype_str = vtype_map.get(v.VType, 'C')
            
            rows.append({
                'var_name': v.VarName,
                'var_type': vtype_str,
                'value': val,
                'fractional_flag': is_frac
            })
            
            if is_frac:
                fractional_count += 1
                fractional_vars.append({
                    'name': v.VarName,
                    'value': val,
                    'distance_to_int': min(val - int(val), 1 - (val - int(val)))
                })
        
        fractional_ratio = fractional_count / total_vars if total_vars > 0 else 0
        
        logger.log(f"Total variables: {total_vars}")
        logger.log(f"Fractional variables: {fractional_count}")
        logger.log(f"Fractional ratio: {fractional_ratio:.4f} ({fractional_ratio*100:.2f}%)")
        
        # Save root_solution.csv
        csv_path = os.path.join(out_dir, 'root_solution.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['var_name', 'var_type', 'value', 'fractional_flag'])
            writer.writeheader()
            writer.writerows(rows)
        logger.log(f"Saved root solution to {csv_path}")
        
        # Top K fractional variables
        fractional_vars.sort(key=lambda x: -abs(x['distance_to_int']))
        top_fractional = fractional_vars[:CONFIG['top_k_fractional']]
        
        if top_fractional:
            logger.log(f"Top {len(top_fractional)} fractional variables:")
            for i, fv in enumerate(top_fractional[:10]):
                logger.log(f"  {i+1}. {fv['name']}: {fv['value']:.6f}")
        
        # Update report
        report['fractional_count'] = fractional_count
        report['fractional_ratio'] = fractional_ratio
        report['top_fractional_vars'] = [
            {'name': fv['name'], 'value': fv['value']} 
            for fv in top_fractional
        ]
        
    except Exception as e:
        logger.log(f"ERROR in Section 2: {str(e)}")
        errors.append(str(e))
        report['fractional_count'] = None
        report['fractional_ratio'] = None
    
    if errors:
        report.setdefault('errors', []).extend(errors)
    
    logger.log("Section 2 complete")

# ============================================================================
# SECTION 3: Structural Properties of the Constraint Matrix
# ============================================================================

def section3_matrix_structure(model, logger: Logger, report: Dict, out_dir: str):
    """
    Provide statistics about coefficient magnitudes, sparsity, and big-M patterns.
    """
    from gurobipy import GRB
    
    logger.log("=" * 60)
    logger.log("SECTION 3: Structural Properties of Constraint Matrix")
    logger.log("=" * 60)
    
    errors = []
    
    try:
        constrs = model.getConstrs()
        variables = model.getVars()
        
        num_rows = len(constrs)
        num_cols = len(variables)
        
        logger.log(f"Number of constraints (rows): {num_rows}")
        logger.log(f"Number of variables (cols): {num_cols}")
        
        # Extract all nonzero coefficients
        coefs = []
        nz_per_row = []
        nz_per_col = defaultdict(int)
        big_M_candidates = []
        
        for i, constr in enumerate(constrs):
            row = model.getRow(constr)
            row_nz = 0
            row_max_coef = 0
            
            for j in range(row.size()):
                coef = abs(row.getCoeff(j))
                var = row.getVar(j)
                
                if coef > 0:
                    coefs.append(coef)
                    row_nz += 1
                    nz_per_col[var.VarName] += 1
                    row_max_coef = max(row_max_coef, coef)
            
            nz_per_row.append(row_nz)
        
        # Also collect objective coefficients
        obj_coefs = []
        for v in variables:
            obj_coefs.append(abs(v.Obj))
        
        total_nonzeros = len(coefs)
        density = total_nonzeros / (num_rows * num_cols) if (num_rows * num_cols) > 0 else 0
        
        logger.log(f"Total nonzeros in A: {total_nonzeros}")
        logger.log(f"Matrix density: {density:.6f} ({density*100:.4f}%)")
        
        # Coefficient statistics
        if coefs:
            coefs_arr = np.array(coefs)
            stats = {
                'min': float(coefs_arr.min()),
                'max': float(coefs_arr.max()),
                'mean': float(coefs_arr.mean()),
                'median': float(np.median(coefs_arr)),
                'std': float(coefs_arr.std()),
                '90pct': float(np.percentile(coefs_arr, 90)),
                '99pct': float(np.percentile(coefs_arr, 99))
            }
            
            logger.log(f"Coefficient stats: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                      f"median={stats['median']:.4f}, mean={stats['mean']:.4f}")
            
            # Big-M detection
            big_M_thresh = max(
                CONFIG['big_M_multiplier'] * stats['median'],
                CONFIG['big_M_absolute_threshold']
            )
            
            for i, constr in enumerate(constrs):
                row = model.getRow(constr)
                for j in range(row.size()):
                    coef = abs(row.getCoeff(j))
                    if coef > big_M_thresh:
                        big_M_candidates.append({
                            'constraint': constr.ConstrName,
                            'coefficient': coef
                        })
                        break  # One per constraint
            
            if big_M_candidates:
                logger.log(f"Big-M candidates detected: {len(big_M_candidates)} constraints")
                for bm in big_M_candidates[:5]:
                    logger.log(f"  - {bm['constraint']}: coef={bm['coefficient']:.2e}")
            else:
                logger.log("No big-M patterns detected")
        else:
            stats = {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, '90pct': 0, '99pct': 0}
        
        # Average nonzeros per row/col
        avg_nz_per_row = np.mean(nz_per_row) if nz_per_row else 0
        avg_nz_per_col = np.mean(list(nz_per_col.values())) if nz_per_col else 0
        
        logger.log(f"Average nonzeros per row: {avg_nz_per_row:.2f}")
        logger.log(f"Average nonzeros per col: {avg_nz_per_col:.2f}")
        
        # Save coefficient stats
        coeff_stats_path = os.path.join(out_dir, 'coeff_stats.json')
        with open(coeff_stats_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'big_M_candidates': [bm['constraint'] for bm in big_M_candidates],
                'big_M_threshold': big_M_thresh if coefs else None
            }, f, indent=2)
        logger.log(f"Saved coefficient stats to {coeff_stats_path}")
        
        # Update report
        report['num_rows'] = num_rows
        report['num_cols'] = num_cols
        report['nonzeros'] = total_nonzeros
        report['density'] = density
        report['avg_nz_per_row'] = avg_nz_per_row
        report['avg_nz_per_col'] = avg_nz_per_col
        report['coeff_stats'] = stats
        report['big_M_candidates'] = [bm['constraint'] for bm in big_M_candidates]
        
    except Exception as e:
        logger.log(f"ERROR in Section 3: {str(e)}")
        errors.append(str(e))
    
    if errors:
        report.setdefault('errors', []).extend(errors)
    
    logger.log("Section 3 complete")

# ============================================================================
# SECTION 4: Symmetry & Repeated Blocks
# ============================================================================

def section4_symmetry(model, logger: Logger, report: Dict):
    """
    Detect obvious symmetry and repeated blocks that might cause branching redundancy.
    """
    from gurobipy import GRB
    
    logger.log("=" * 60)
    logger.log("SECTION 4: Symmetry & Repeated Blocks")
    logger.log("=" * 60)
    
    errors = []
    
    try:
        variables = model.getVars()
        constrs = model.getConstrs()
        
        hash_tol = CONFIG['hash_tolerance']
        min_group_size = CONFIG['min_symmetric_group_size']
        
        # Build column vectors for each variable
        # Column vector = (objective coef, [(constr_idx, coef), ...])
        column_data = {}
        
        for v in variables:
            col = model.getCol(v)
            # Get constraint participation
            col_entries = []
            for i in range(col.size()):
                constr = col.getConstr(i)
                coef = col.getCoeff(i)
                col_entries.append((constr.index, round(coef / hash_tol) * hash_tol))
            
            col_entries.sort()  # Sort by constraint index for consistent hashing
            
            # Create hashable representation
            obj_rounded = round(v.Obj / hash_tol) * hash_tol
            col_tuple = (obj_rounded, tuple(col_entries))
            col_hash = hashlib.md5(str(col_tuple).encode()).hexdigest()
            
            column_data[v.VarName] = {
                'hash': col_hash,
                'obj': v.Obj,
                'col_tuple': col_tuple
            }
        
        # Group by hash
        hash_groups = defaultdict(list)
        for var_name, data in column_data.items():
            hash_groups[data['hash']].append(var_name)
        
        # Find symmetric groups (size >= min_group_size)
        symmetric_groups = []
        for hash_val, var_names in hash_groups.items():
            if len(var_names) >= min_group_size:
                symmetric_groups.append({
                    'size': len(var_names),
                    'repr_vars': var_names[:5],  # First 5 as representatives
                    'pattern_hash': hash_val[:12],  # Truncated hash
                    'all_vars': var_names
                })
        
        # Sort by size descending
        symmetric_groups.sort(key=lambda x: -x['size'])
        
        num_symmetric_groups = len(symmetric_groups)
        total_symmetric_vars = sum(g['size'] for g in symmetric_groups)
        
        logger.log(f"Total variables: {len(variables)}")
        logger.log(f"Symmetric groups found: {num_symmetric_groups}")
        logger.log(f"Variables in symmetric groups: {total_symmetric_vars}")
        
        if symmetric_groups:
            logger.log("Top symmetric groups:")
            for i, g in enumerate(symmetric_groups[:10]):
                logger.log(f"  {i+1}. Size={g['size']}, hash={g['pattern_hash']}, "
                          f"vars={g['repr_vars'][:3]}...")
            
            # Suggest remedies
            logger.log("\nRemedies for symmetry:")
            logger.log("  - Add ordering constraints (e.g., x_1 >= x_2 >= ... >= x_n)")
            logger.log("  - Use symmetry-breaking inequalities")
            logger.log("  - Consider aggregating symmetric variables where feasible")
            logger.log("  - Enable solver symmetry detection (Gurobi: Symmetry parameter)")
        
        # Check for identical rows (repeated constraints)
        row_hashes = defaultdict(list)
        for constr in constrs:
            row = model.getRow(constr)
            row_entries = []
            for j in range(row.size()):
                var = row.getVar(j)
                coef = row.getCoeff(j)
                row_entries.append((var.index, round(coef / hash_tol) * hash_tol))
            row_entries.sort()
            
            # Include RHS and sense
            rhs_rounded = round(constr.RHS / hash_tol) * hash_tol
            row_tuple = (constr.Sense, rhs_rounded, tuple(row_entries))
            row_hash = hashlib.md5(str(row_tuple).encode()).hexdigest()
            
            row_hashes[row_hash].append(constr.ConstrName)
        
        identical_row_groups = [
            {'size': len(names), 'constraints': names[:5]}
            for names in row_hashes.values() if len(names) >= 2
        ]
        
        if identical_row_groups:
            logger.log(f"\nIdentical constraint groups: {len(identical_row_groups)}")
            for i, g in enumerate(identical_row_groups[:5]):
                logger.log(f"  {i+1}. Size={g['size']}, constrs={g['constraints'][:3]}...")
        
        # Update report
        report['num_symmetric_groups'] = num_symmetric_groups
        report['symmetric_groups'] = [
            {'size': g['size'], 'repr_vars': g['repr_vars'], 'pattern_hash': g['pattern_hash']}
            for g in symmetric_groups[:20]  # Limit to top 20
        ]
        report['total_symmetric_vars'] = total_symmetric_vars
        report['identical_row_groups'] = len(identical_row_groups)
        
    except Exception as e:
        logger.log(f"ERROR in Section 4: {str(e)}")
        errors.append(str(e))
        report['num_symmetric_groups'] = None
        report['symmetric_groups'] = []
    
    if errors:
        report.setdefault('errors', []).extend(errors)
    
    logger.log("Section 4 complete")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_diagnostics(input_path: str, out_dir: str, time_limit: int = 300):
    """
    Run all four diagnostic sections on an ILP file.
    """
    import gurobipy as gp
    
    # Setup
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(os.path.join(out_dir, 'log.txt'))
    report = {
        'input_file': input_path,
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG
    }
    
    logger.log("=" * 60)
    logger.log("ILP HARDNESS DIAGNOSTIC TOOL")
    logger.log("=" * 60)
    logger.log(f"Input file: {input_path}")
    logger.log(f"Output directory: {out_dir}")
    logger.log(f"Time limit: {time_limit}s")
    logger.log("")
    
    CONFIG['mip_time_limit'] = time_limit
    
    try:
        # Load model
        logger.log("Loading model...")
        model = gp.read(input_path)
        logger.log(f"Model loaded: {model.ModelName}")
        logger.log(f"  Variables: {model.NumVars}")
        logger.log(f"  Constraints: {model.NumConstrs}")
        logger.log(f"  Binary vars: {model.NumBinVars}")
        logger.log(f"  Integer vars: {model.NumIntVars}")
        logger.log(f"  Continuous vars: {model.NumVars - model.NumBinVars - model.NumIntVars}")
        logger.log("")
        
        report['model_name'] = model.ModelName
        report['num_vars'] = model.NumVars
        report['num_constrs'] = model.NumConstrs
        report['num_bin_vars'] = model.NumBinVars
        report['num_int_vars'] = model.NumIntVars
        
        # Run sections
        rel, lp_obj = section1_integrality_gap(model, logger, report)
        logger.log("")
        
        section2_fractionality(rel, logger, report, out_dir)
        logger.log("")
        
        section3_matrix_structure(model, logger, report, out_dir)
        logger.log("")
        
        section4_symmetry(model, logger, report)
        logger.log("")
        
    except Exception as e:
        logger.log(f"FATAL ERROR: {str(e)}")
        report['errors'] = report.get('errors', []) + [f"Fatal: {str(e)}"]
    
    # Summary
    logger.log("=" * 60)
    logger.log("DIAGNOSTIC SUMMARY")
    logger.log("=" * 60)
    
    # Hardness assessment
    hardness_score = 0
    hardness_factors = []
    
    if report.get('integrality_gap') is not None:
        gap = report['integrality_gap']
        if gap > 0.5:
            hardness_score += 3
            hardness_factors.append(f"Large integrality gap ({gap:.2%})")
        elif gap > 0.1:
            hardness_score += 2
            hardness_factors.append(f"Moderate integrality gap ({gap:.2%})")
        elif gap > 0.01:
            hardness_score += 1
            hardness_factors.append(f"Small integrality gap ({gap:.2%})")
    
    if report.get('fractional_ratio') is not None:
        frac = report['fractional_ratio']
        if frac > 0.5:
            hardness_score += 3
            hardness_factors.append(f"High fractionality ({frac:.2%})")
        elif frac > 0.2:
            hardness_score += 2
            hardness_factors.append(f"Moderate fractionality ({frac:.2%})")
        elif frac > 0.05:
            hardness_score += 1
            hardness_factors.append(f"Low fractionality ({frac:.2%})")
    
    if report.get('big_M_candidates'):
        n_bigm = len(report['big_M_candidates'])
        if n_bigm > 10:
            hardness_score += 2
            hardness_factors.append(f"Many big-M constraints ({n_bigm})")
        elif n_bigm > 0:
            hardness_score += 1
            hardness_factors.append(f"Some big-M constraints ({n_bigm})")
    
    if report.get('num_symmetric_groups') is not None:
        n_sym = report['num_symmetric_groups']
        if n_sym > 10:
            hardness_score += 3
            hardness_factors.append(f"High symmetry ({n_sym} groups)")
        elif n_sym > 3:
            hardness_score += 2
            hardness_factors.append(f"Moderate symmetry ({n_sym} groups)")
        elif n_sym > 0:
            hardness_score += 1
            hardness_factors.append(f"Low symmetry ({n_sym} groups)")
    
    if hardness_score >= 8:
        hardness_level = "VERY HARD"
    elif hardness_score >= 5:
        hardness_level = "HARD"
    elif hardness_score >= 3:
        hardness_level = "MODERATE"
    else:
        hardness_level = "EASY"
    
    report['hardness_score'] = hardness_score
    report['hardness_level'] = hardness_level
    report['hardness_factors'] = hardness_factors
    
    logger.log(f"Hardness Level: {hardness_level} (score: {hardness_score})")
    logger.log("Contributing factors:")
    for factor in hardness_factors:
        logger.log(f"  - {factor}")
    
    # Save report
    report_path = os.path.join(out_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.log(f"\nReport saved to: {report_path}")
    
    # Save log
    logger.save()
    logger.log(f"Log saved to: {os.path.join(out_dir, 'log.txt')}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='ILP Hardness Diagnostic Tool')
    parser.add_argument('input_path', help='Path to ILP file (LP, MPS, or model)')
    parser.add_argument('--out_dir', default='hardness_output', 
                       help='Output directory for reports')
    parser.add_argument('--time_limit', type=int, default=300,
                       help='Time limit for MIP solve (seconds)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)
    
    run_diagnostics(args.input_path, args.out_dir, args.time_limit)


if __name__ == '__main__':
    main()
