import numpy as np
from typing import List, Set, Tuple
from utils.bitset import BitSetOperations
from algorithms.next_closure import NextClosure

set_operations = BitSetOperations()

class ConceptLattice:
    def __init__(self,extent_intent, intent_extent, objects_, attributes_, min_support=None):
        """
            Args:
                extent_intent: a list where the index indicates the index of an object
                                and the binary representation of the value indicates the 
                                intents held by the object
                intent_extent: same as the extent_intent with roles reversed

                objects_: list of labeled objects

                attributes: list of labeled attributes
        """
        self.objects = objects_
        self.attributes = attributes_
        self.num_objects = len(self.objects)
        self.num_attributes = len(self.attributes)

        # Mappings
        self.obj_to_index = {obj: i for i, obj in enumerate(self.objects)}
        self.attr_to_index = {attr: i for i, attr in enumerate(self.attributes)}

        # Compute bit-columns for attributes and bit-rows for objects
        self.attribute_bit_columns = intent_extent
        self.object_bit_rows = extent_intent

        self.min_support = min_support



    def all_concepts(self) -> List[Tuple]:
        """
            Computes all concepts in the formal context.
            Args:
                None
            Returns:
                List of tuples where each tuple contains a set of objects and a set of attributes.
        """
        if self.min_support:
            next_closure = NextClosure(self.num_objects, self.num_attributes, self.object_bit_rows, self.attribute_bit_columns, self.min_support)
            return next_closure.all_pruned_concepts()
        else:
            next_closure = NextClosure(self.num_objects, self.num_attributes, self.object_bit_rows, self.attribute_bit_columns)
            return next_closure.all_concepts()
