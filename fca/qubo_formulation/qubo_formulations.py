import numpy as np
import networkx as nx

class QuboFormulation:
    """
    A class to represent the QUBO formulation for different problems in FCA.
    """

    def __init__(self):
        pass

    def build_iceberg_qubo(self, concepts, min_support, alpha=1.0, lambda_overlap=1.0, lambda_hierarchy=2.0, lattice_graph=None):
        """
        Build a structured QUBO matrix for Iceberg concept selection from a concept lattice.
        
        Args:
            concepts: List of (extent, intent) pairs
            min_support: Minimum extent size to qualify as iceberg
            alpha: Reward multiplier for iceberg concepts
            lambda_overlap: Penalty multiplier for overlapping extents between selected concepts
            lambda_hierarchy: Penalty for violating parent-child selection logic (requires lattice_graph)
            lattice_graph: networkx.DiGraph of concept lattice (optional, for hierarchy enforcement)
            
        Returns:
            Q: QUBO matrix as numpy array
        """
        n = len(concepts)
        Q = np.zeros((n, n))

        # Support-based diagonal values
        for i in range(n):
            support = concepts[i][0].bit_count() 
            # Reward or penalty based on support                    
            if support < min_support:
                Q[i][i] += (min_support - support) ** 2  # penalty
            else:
                Q[i][i] += -alpha * support              # reward

        # Overlap penalty (off-diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                overlap = (concepts[i][0] & concepts[j][0]).bit_count()
                if overlap > 0:
                    Q[i][j] += lambda_overlap * overlap
                    Q[j][i] += lambda_overlap * overlap

        # Parent-child constraint (x_child <= x_parent → penalty for child without parent)
        if lattice_graph is not None:
            for parent, child in lattice_graph.edges():
                i = list(concepts).index(parent)
                j = list(concepts).index(child)
                # Penalize (x_child - x_parent)^2 = x_child^2 + x_parent^2 - 2 x_child x_parent
                Q[i][i] += lambda_hierarchy
                Q[j][j] += lambda_hierarchy
                Q[i][j] += -2 * lambda_hierarchy
                Q[j][i] += -2 * lambda_hierarchy

        return Q

    def build_qubo(self, candidates, context, alpha=0.5, beta=1.0, k_target=None):
        """
        candidates: list with keys 'gain' and 'extent_idx'
        context: list of rows (for computing overlap)
        returns Q (numpy 2D array), linear bias h, constant offset
        """
        
        m = len(candidates)
        # linear terms: reward = gain
        linear = np.array([c['gini_gain'] for key, c in candidates.items()], dtype=float)

        # pairwise overlap penalty: J_ij = overlap size / n
        n = len(context)
        overlaps = np.zeros((m, m), dtype=float)
        
        Q = np.zeros((m, m), dtype=float)


        candidate_data = list(candidates.keys())
        for i in range(m):
            ext_i = candidate_data[i][0]
            for j in range(i+1, m):
                ext_j = candidate_data[j][0]
                ov = (ext_i & ext_j).bit_count() / n if n>0 else 0.0
                overlaps[i, j] = ov
                overlaps[j, i] = ov
        # Linear contribution (diagonal)
        for i in range(m):
            Q[i, i] += -linear[i] + beta * (1.0)  # -gain + beta*1 from x_i^2 term
        # Off-diagonals from overlap penalty and quadratic penalty
        for i in range(m):
            for j in range(i+1, m):
                Q[i, j] += alpha * overlaps[i, j] + beta * 1.0 
                Q[j, i] = Q[i, j]
        # Also linear has -2*beta*k term from expansion (we add to diagonal as linear bias)
        offset = 0.0
        if k_target is not None:
            for i in range(m):
                Q[i, i] += -2.0 * beta * k_target 
            offset += beta * (k_target ** 2)
        return Q, offset

