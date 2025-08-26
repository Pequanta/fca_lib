from algorithms.next_closure import NextClosure
import networkx as nx
from typing import List, Tuple
from fca.utils.utils import count_ones, gini_gain_over_baseline, gini_impurity_from_counts
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

    def get_iceberg_data(
            self,
            concept,
            goal_attr,
            num_objects,
            baseline_counts,
            class_counts = None
        ):
        index = 0
        if class_counts is None:
            class_counts = {0: 0, 1: 0}
            while index < num_objects:
                if (1 << index) & concept[0] != 0:
                    if (1 << index) & goal_attr[index] != 0:
                        class_counts[1] += 1
                    else:
                        class_counts[0] += 1
                index += 1

        result = {
            "support": (count_ones(concept[0]) / num_objects),
            "class_counts": class_counts,
            "gini_impurity":  gini_impurity_from_counts(class_counts),
            "gini_gain": gini_gain_over_baseline(class_counts, baseline_counts)
        }

        return result

        


