import numpy as np
import networkx as nx
from utils.tools import count_ones

class QuboFormulation:
    """
    A class to represent the QUBO formulation for different problems in FCA.
    """

    def __init__(self):
        pass

    @staticmethod
    def build_iceberg_qubo(concepts, min_support, alpha=1.0, lambda_overlap=1.0, lambda_hierarchy=2.0, lattice_graph=None):
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
        

        # 1. Support-based diagonal values
        for i in range(n):
            support = count_ones(concepts[i][0]) 
            # Reward or penalty based on support                    
            if support < min_support:
                Q[i][i] += (min_support - support) ** 2  # penalty
            else:
                Q[i][i] += -alpha * support              # reward

        # 2. Overlap penalty (off-diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                overlap = count_ones(concepts[i][0] & concepts[j][0])
                if overlap > 0:
                    Q[i][j] += lambda_overlap * overlap
                    Q[j][i] += lambda_overlap * overlap

        # 3. Parent-child constraint (x_child <= x_parent → penalty for child without parent)
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
