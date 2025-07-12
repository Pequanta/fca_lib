from algorithms.next_closure import NextClosure
import networkx as nx
from typing import List, Tuple
from utils.tools import count_ones
class IcebergConcept:
    """
    A class to represent an iceberg concept in a formal context.
    An iceberg concept is a concept that has a support above a certain threshold.
    """
    
    def __init__(self):
        """
        Args:
            num_objects (int): Number of objects in the context.
            num_attributes (int): Number of attributes in the context.
            object_bit_rows (list[int]): Bitset representation of objects.
            attribute_bit_columns (list[int]): Bitset representation of attributes.
            threshold (float): Support threshold for the iceberg concept.
        """
        pass
    
    def extract_iceberg_concepts(self, concepts: List[Tuple[int]], min_support: int):
        """
        Filters concepts from a concept lattice graph based on the iceberg condition.

        Parameters:
            concept_lattice (nx.DiGraph): A concept lattice with 'extent' attribute per node.
            min_support (int): Minimum number of objects required in extent.

        Returns:
            List of tuples: [(node_id, extent, intent), ...] satisfying the min_support condition.
        """
        iceberg_concepts = []
        for node in concepts:
            if count_ones(node[0]) >= min_support:
                iceberg_concepts.append(node)
        return iceberg_concepts
            