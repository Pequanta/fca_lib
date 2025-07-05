import numpy as np
from typing import List, Set, Tuple
from utils.bitset import BitSetOperations


set_operations = BitSetOperations()

class ConceptLattice:
    def __init__(self,extent_intent, intent_extent, objects_, attributes_):
        """
        :param context: dict mapping object name → set of attribute names
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



    def _intent_closure_bitset(self, bitset: int) -> int:
        """
        Closure of an intent (attribute bitset) → intent (attribute bitset)
        """
        # Get common objects
        extent_bits = (1 << self.num_objects) - 1
        for i in range(self.num_attributes):
            if bitset & (1 << i):
                extent_bits &= self.attribute_bit_columns[i]

        # Intersect all attributes of the objects in the extent
        closed_intent = (1 << self.num_attributes) - 1
        for obj_idx in range(self.num_objects):
            if extent_bits & (1 << obj_idx):
                closed_intent &= self.object_bit_rows[obj_idx]

        return closed_intent

    def _next_closure_bitset(self, A: int) -> int | None:
        """
        Computes the next lectically closed attribute set after A (bitset)
        """
        n = self.num_attributes
        for i in reversed(range(n)):
            mask = 1 << i
            if A & mask:
                A &= ~mask
            else:
                candidate = A | mask
                closed = self._intent_closure_bitset(candidate)
                if (closed & ~A) & ((1 << i) - 1) == 0:
                    return closed
        return None


    
    def all_concepts(self) -> list[tuple[set[str], set[str]]]:
        concepts = []
        seen = set()
        current = 0

        while current is not None:
            closed = self._intent_closure_bitset(current)
            if closed not in seen:
                seen.add(closed)

                extent_bits = (1 << self.num_objects) - 1
                for i in range(self.num_attributes):
                    if closed & (1 << i):
                        extent_bits &= self.attribute_bit_columns[i]

                concepts.append((extent_bits, closed))

            current = self._next_closure_bitset(current)

        return concepts


