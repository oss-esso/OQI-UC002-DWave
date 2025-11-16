"""
SAT Instance Hardness Metrics and Complexity Analysis

Implements various metrics to characterize SAT instance hardness and complexity,
allowing comparison between real-world conservation instances and random k-SAT
instances from QAOA benchmarks.

Metrics include:
- Clause-to-variable ratio (α)
- Variable-clause graph properties
- Backbone size
- Solution space density
- Constraint tightness
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from collections import Counter

try:
    from pysat.solvers import Glucose4
    PYSAT_AVAILABLE = True
except ImportError:
    PYSAT_AVAILABLE = False


@dataclass
class HardnessMetrics:
    """
    Comprehensive hardness metrics for SAT instances
    """
    # Basic statistics
    n_variables: int
    n_clauses: int
    k_avg: float  # Average clause size
    alpha: float  # Clause-to-variable ratio (m/n)
    
    # Graph properties
    vcg_density: float  # Variable-clause graph density
    vcg_avg_degree: float  # Average degree in VCG
    vcg_clustering: float  # Clustering coefficient
    
    # Constraint properties
    positive_negative_ratio: float  # Ratio of positive to negative literals
    literal_frequency_entropy: float  # Shannon entropy of literal frequency
    clause_overlap: float  # Average variables shared between clauses
    
    # Solution space (if computable)
    backbone_size: Optional[int] = None  # Number of backbone variables
    backbone_fraction: Optional[float] = None
    solution_density_estimate: Optional[float] = None
    
    # Complexity indicators
    hardness_score: Optional[float] = None  # Combined hardness score
    expected_difficulty: str = "unknown"  # 'easy', 'medium', 'hard', 'very_hard'


def compute_hardness_metrics(
    n: int,
    clauses: List[List[int]],
    sample_for_backbone: int = 100
) -> HardnessMetrics:
    """
    Compute comprehensive hardness metrics for a SAT instance
    
    Args:
        n: Number of variables
        clauses: List of clauses (each clause is list of literals)
        sample_for_backbone: Number of random assignments to sample for backbone estimation
    
    Returns:
        HardnessMetrics object
    """
    m = len(clauses)
    
    # Basic statistics
    k_avg = np.mean([len(clause) for clause in clauses])
    alpha = m / n if n > 0 else 0
    
    # Variable-clause graph properties
    vcg_metrics = _compute_vcg_properties(n, clauses)
    
    # Constraint properties
    constraint_metrics = _compute_constraint_properties(n, clauses)
    
    # Backbone and solution space (if PySAT available)
    backbone_metrics = _estimate_backbone_and_solutions(
        n, clauses, sample_for_backbone
    ) if PYSAT_AVAILABLE else {}
    
    # Compute combined hardness score
    hardness_score = _compute_hardness_score(
        alpha, vcg_metrics, constraint_metrics
    )
    
    # Classify difficulty
    expected_difficulty = _classify_difficulty(hardness_score, alpha, n)
    
    return HardnessMetrics(
        n_variables=n,
        n_clauses=m,
        k_avg=k_avg,
        alpha=alpha,
        vcg_density=vcg_metrics['density'],
        vcg_avg_degree=vcg_metrics['avg_degree'],
        vcg_clustering=vcg_metrics['clustering'],
        positive_negative_ratio=constraint_metrics['pos_neg_ratio'],
        literal_frequency_entropy=constraint_metrics['literal_entropy'],
        clause_overlap=constraint_metrics['clause_overlap'],
        backbone_size=backbone_metrics.get('backbone_size'),
        backbone_fraction=backbone_metrics.get('backbone_fraction'),
        solution_density_estimate=backbone_metrics.get('solution_density'),
        hardness_score=hardness_score,
        expected_difficulty=expected_difficulty
    )


def _compute_vcg_properties(n: int, clauses: List[List[int]]) -> Dict:
    """
    Compute Variable-Clause Graph (VCG) properties
    
    VCG is a bipartite graph where:
    - One set: variables
    - Other set: clauses
    - Edge (v, c) if variable v appears in clause c
    """
    # Build bipartite graph
    G = nx.Graph()
    
    # Add variable nodes
    var_nodes = [f"v{i}" for i in range(1, n+1)]
    G.add_nodes_from(var_nodes, bipartite=0)
    
    # Add clause nodes and edges
    for c_idx, clause in enumerate(clauses):
        clause_node = f"c{c_idx}"
        G.add_node(clause_node, bipartite=1)
        
        for lit in clause:
            var = abs(lit)
            var_node = f"v{var}"
            G.add_edge(var_node, clause_node)
    
    # Compute properties
    if G.number_of_edges() > 0:
        # Density
        max_edges = n * len(clauses)
        density = G.number_of_edges() / max_edges if max_edges > 0 else 0
        
        # Average degree
        degrees = [G.degree(node) for node in var_nodes]
        avg_degree = np.mean(degrees) if degrees else 0
        
        # Clustering coefficient (only meaningful for variable nodes)
        clustering = nx.average_clustering(G)
    else:
        density = 0
        avg_degree = 0
        clustering = 0
    
    return {
        'density': density,
        'avg_degree': avg_degree,
        'clustering': clustering
    }


def _compute_constraint_properties(n: int, clauses: List[List[int]]) -> Dict:
    """Compute constraint-related properties"""
    
    # Literal frequency
    literal_counts = Counter()
    for clause in clauses:
        for lit in clause:
            literal_counts[lit] += 1
    
    # Positive vs negative literals
    pos_count = sum(count for lit, count in literal_counts.items() if lit > 0)
    neg_count = sum(count for lit, count in literal_counts.items() if lit < 0)
    pos_neg_ratio = pos_count / neg_count if neg_count > 0 else float('inf')
    
    # Literal frequency entropy (how balanced are literal occurrences)
    frequencies = np.array(list(literal_counts.values()))
    if len(frequencies) > 0:
        probs = frequencies / np.sum(frequencies)
        literal_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        literal_entropy = 0
    
    # Clause overlap (average shared variables between clauses)
    overlaps = []
    for i in range(len(clauses)):
        for j in range(i+1, len(clauses)):
            vars_i = set(abs(lit) for lit in clauses[i])
            vars_j = set(abs(lit) for lit in clauses[j])
            overlap = len(vars_i & vars_j)
            overlaps.append(overlap)
    
    clause_overlap = np.mean(overlaps) if overlaps else 0
    
    return {
        'pos_neg_ratio': pos_neg_ratio,
        'literal_entropy': literal_entropy,
        'clause_overlap': clause_overlap
    }


def _estimate_backbone_and_solutions(
    n: int,
    clauses: List[List[int]],
    num_samples: int = 100
) -> Dict:
    """
    Estimate backbone variables and solution density
    
    Backbone: Variables that have the same value in all satisfying assignments
    Solution density: Estimated fraction of satisfying assignments
    
    This uses sampling, so results are estimates.
    """
    if not PYSAT_AVAILABLE:
        return {}
    
    try:
        # Try to find solutions with random assumptions
        solver = Glucose4()
        for clause in clauses:
            solver.add_clause(clause)
        
        # Sample random complete assignments
        satisfying_assignments = []
        for _ in range(num_samples):
            # Random assumptions
            assumptions = [
                i if np.random.random() < 0.5 else -i 
                for i in range(1, n+1)
            ]
            
            if solver.solve(assumptions=assumptions):
                model = solver.get_model()
                satisfying_assignments.append(model)
                
            if len(satisfying_assignments) >= 20:  # Found enough
                break
        
        solver.delete()
        
        if len(satisfying_assignments) == 0:
            # Instance might be UNSAT or very hard
            return {
                'backbone_size': None,
                'backbone_fraction': None,
                'solution_density': 0.0
            }
        
        # Estimate backbone
        # A variable is in backbone if it has same value in all sampled solutions
        backbone_vars = []
        for var in range(1, n+1):
            values = [model[var-1] > 0 for model in satisfying_assignments]
            if all(values) or not any(values):  # All True or all False
                backbone_vars.append(var)
        
        backbone_size = len(backbone_vars)
        backbone_fraction = backbone_size / n
        
        # Estimate solution density (very rough)
        # If we found k solutions out of m samples, estimate density
        solution_density = len(satisfying_assignments) / num_samples
        
        return {
            'backbone_size': backbone_size,
            'backbone_fraction': backbone_fraction,
            'solution_density': solution_density
        }
        
    except Exception as e:
        # Solver failed
        return {
            'backbone_size': None,
            'backbone_fraction': None,
            'solution_density': None
        }


def _compute_hardness_score(
    alpha: float,
    vcg_metrics: Dict,
    constraint_metrics: Dict
) -> float:
    """
    Compute combined hardness score (0-100 scale)
    
    Combines multiple indicators:
    - α distance from phase transition (higher = harder near transition)
    - VCG density and clustering (more connected = harder)
    - Constraint balance (balanced pos/neg = harder)
    """
    score = 0.0
    
    # Alpha contribution (30 points max)
    # Hardest near phase transition (α ≈ 4.27 for 3-SAT)
    alpha_optimal = 4.27
    alpha_distance = abs(alpha - alpha_optimal)
    alpha_score = 30 * np.exp(-alpha_distance / 2.0)
    score += alpha_score
    
    # VCG density (25 points max)
    # Higher density = more constrained = harder
    density_score = 25 * vcg_metrics['density']
    score += density_score
    
    # VCG clustering (20 points max)
    # Higher clustering = more structure = can be easier OR harder
    clustering_score = 20 * vcg_metrics['clustering']
    score += clustering_score
    
    # Literal balance (25 points max)
    # Ratio close to 1.0 = balanced = harder
    pos_neg_ratio = constraint_metrics['pos_neg_ratio']
    if pos_neg_ratio < 1.0:
        balance = pos_neg_ratio
    else:
        balance = 1.0 / pos_neg_ratio if pos_neg_ratio > 0 else 0
    balance_score = 25 * balance
    score += balance_score
    
    return min(score, 100.0)


def _classify_difficulty(hardness_score: float, alpha: float, n: int) -> str:
    """Classify instance difficulty"""
    
    # Small instances are generally easier
    if n < 20:
        if hardness_score < 40:
            return "easy"
        elif hardness_score < 60:
            return "medium"
        else:
            return "hard"
    
    # Larger instances
    if hardness_score < 30:
        return "easy"
    elif hardness_score < 50:
        return "medium"
    elif hardness_score < 70:
        return "hard"
    else:
        return "very_hard"


def compare_instances(
    instance1_metrics: HardnessMetrics,
    instance2_metrics: HardnessMetrics,
    name1: str = "Instance 1",
    name2: str = "Instance 2"
) -> str:
    """
    Generate comparison report between two instances
    
    Returns:
        Formatted comparison string
    """
    report = []
    report.append("=" * 70)
    report.append("INSTANCE COMPARISON")
    report.append("=" * 70)
    
    report.append(f"\n{name1}:")
    report.append(f"  Variables: {instance1_metrics.n_variables}")
    report.append(f"  Clauses: {instance1_metrics.n_clauses}")
    report.append(f"  α (m/n): {instance1_metrics.alpha:.3f}")
    report.append(f"  Hardness Score: {instance1_metrics.hardness_score:.1f}/100")
    report.append(f"  Expected Difficulty: {instance1_metrics.expected_difficulty}")
    
    report.append(f"\n{name2}:")
    report.append(f"  Variables: {instance2_metrics.n_variables}")
    report.append(f"  Clauses: {instance2_metrics.n_clauses}")
    report.append(f"  α (m/n): {instance2_metrics.alpha:.3f}")
    report.append(f"  Hardness Score: {instance2_metrics.hardness_score:.1f}/100")
    report.append(f"  Expected Difficulty: {instance2_metrics.expected_difficulty}")
    
    report.append("\nDetailed Comparison:")
    report.append(f"  VCG Density:        {instance1_metrics.vcg_density:.3f} vs {instance2_metrics.vcg_density:.3f}")
    report.append(f"  VCG Clustering:     {instance1_metrics.vcg_clustering:.3f} vs {instance2_metrics.vcg_clustering:.3f}")
    report.append(f"  Pos/Neg Ratio:      {instance1_metrics.positive_negative_ratio:.3f} vs {instance2_metrics.positive_negative_ratio:.3f}")
    report.append(f"  Literal Entropy:    {instance1_metrics.literal_frequency_entropy:.3f} vs {instance2_metrics.literal_frequency_entropy:.3f}")
    report.append(f"  Clause Overlap:     {instance1_metrics.clause_overlap:.3f} vs {instance2_metrics.clause_overlap:.3f}")
    
    if instance1_metrics.backbone_fraction is not None and instance2_metrics.backbone_fraction is not None:
        report.append(f"  Backbone Fraction:  {instance1_metrics.backbone_fraction:.3f} vs {instance2_metrics.backbone_fraction:.3f}")
    
    # Similarity score
    similarity = compute_similarity_score(instance1_metrics, instance2_metrics)
    report.append(f"\nSimilarity Score: {similarity:.1f}/100 (100 = identical structure)")
    
    return "\n".join(report)


def compute_similarity_score(
    metrics1: HardnessMetrics,
    metrics2: HardnessMetrics
) -> float:
    """
    Compute structural similarity between two instances (0-100 scale)
    """
    score = 0.0
    
    # Alpha similarity (30 points)
    alpha_diff = abs(metrics1.alpha - metrics2.alpha)
    alpha_sim = 30 * np.exp(-alpha_diff)
    score += alpha_sim
    
    # VCG structure similarity (40 points)
    density_diff = abs(metrics1.vcg_density - metrics2.vcg_density)
    clustering_diff = abs(metrics1.vcg_clustering - metrics2.vcg_clustering)
    vcg_sim = 20 * np.exp(-density_diff * 2) + 20 * np.exp(-clustering_diff * 2)
    score += vcg_sim
    
    # Constraint similarity (30 points)
    pos_neg_diff = abs(np.log(metrics1.positive_negative_ratio + 1e-10) - 
                       np.log(metrics2.positive_negative_ratio + 1e-10))
    entropy_diff = abs(metrics1.literal_frequency_entropy - metrics2.literal_frequency_entropy)
    constraint_sim = 15 * np.exp(-pos_neg_diff / 2) + 15 * np.exp(-entropy_diff / 2)
    score += constraint_sim
    
    return min(score, 100.0)


if __name__ == "__main__":
    print("=" * 70)
    print("SAT HARDNESS METRICS DEMO")
    print("=" * 70)
    
    # Example 1: Random 3-SAT
    print("\n1. Random 3-SAT Instance (n=30, α=4.27)")
    clauses_random = [
        [1, -2, 3], [-1, 2, 4], [1, -3, -4],
        [2, 3, -5], [-2, 4, 5], [1, -4, 5],
        [3, -5, 6], [-3, 4, -6], [2, 5, -6],
        # ... (truncated for demo)
    ]
    n_random = 6
    clauses_random = []
    import random
    random.seed(42)
    for _ in range(int(4.27 * 30)):
        vars = random.sample(range(1, 31), 3)
        clause = [v if random.random() < 0.5 else -v for v in vars]
        clauses_random.append(clause)
    
    metrics_random = compute_hardness_metrics(30, clauses_random)
    print(f"   Variables: {metrics_random.n_variables}")
    print(f"   Clauses: {metrics_random.n_clauses}")
    print(f"   α: {metrics_random.alpha:.3f}")
    print(f"   VCG Density: {metrics_random.vcg_density:.4f}")
    print(f"   VCG Clustering: {metrics_random.vcg_clustering:.4f}")
    print(f"   Hardness Score: {metrics_random.hardness_score:.1f}/100")
    print(f"   Expected Difficulty: {metrics_random.expected_difficulty}")
    
    # Example 2: Structured instance
    print("\n2. Structured Instance (Conservation Problem)")
    # Simulate structured clauses (more regular pattern)
    clauses_structured = []
    for i in range(1, 26, 2):
        clauses_structured.append([i, i+1, i+2])
        clauses_structured.append([-i, i+1, -i+2])
        clauses_structured.append([i, -i+1, i+2])
    
    metrics_structured = compute_hardness_metrics(28, clauses_structured)
    print(f"   Variables: {metrics_structured.n_variables}")
    print(f"   Clauses: {metrics_structured.n_clauses}")
    print(f"   α: {metrics_structured.alpha:.3f}")
    print(f"   VCG Density: {metrics_structured.vcg_density:.4f}")
    print(f"   VCG Clustering: {metrics_structured.vcg_clustering:.4f}")
    print(f"   Hardness Score: {metrics_structured.hardness_score:.1f}/100")
    print(f"   Expected Difficulty: {metrics_structured.expected_difficulty}")
    
    # Example 3: Comparison
    print("\n3. Comparison")
    print(compare_instances(
        metrics_random, metrics_structured,
        "Random 3-SAT", "Structured Instance"
    ))
    
    print("\n✓ Hardness metrics computed successfully")
    if not PYSAT_AVAILABLE:
        print("\n⚠ Note: PySAT not available, backbone estimation disabled")
        print("   Install with: pip install python-sat")
