from fca.utils.utils import count_ones
class NextClosure:
    def __init__(self, num_objects: int, num_attributes: int, object_bit_rows: list[int], attribute_bit_columns: list[int], min_support= None):
        """
        Args:
            num_objects (int): Number of objects in the context.
            num_attributes (int): Number of attributes in the context.
            object_bit_rows (list[int]): Bitset representation of objects.
            attribute_bit_columns (list[int]): Bitset representation of attributes.
        """
        self.num_objects = num_objects
        self.num_attributes = num_attributes
        self.object_bit_rows = object_bit_rows
        self.attribute_bit_columns = attribute_bit_columns
        self.min_support = min_support

    def _intent_closure_bitset(self, bitset: int) -> int:
        """
        Args:
            pass
        Returns:
            pass    
        
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
        Computes the next lectically closed attribute set after A (bitset)x
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
    

    def _support(self, extent_bits: int) -> float:
        """Support = fraction of objects covered by this extent."""
        return count_ones(extent_bits) / self.num_objects if self.num_objects > 0 else 0
    def _compute_extent(self, intent_bits: int) -> int:
        """Helper: compute extent for a given intent."""
        extent_bits = (1 << self.num_objects) - 1
        for i in range(self.num_attributes):
            if intent_bits & (1 << i):
                extent_bits &= self.attribute_bit_columns[i]
        return extent_bits
    def all_pruned_concepts(self) -> list[tuple[int, int]]:
        """
            Generate concepts (extent_bits, intent_bits), pruned by Titanic-style iceberg condition.
        """
        concepts = []
        seen = set()
        current = 0

        while current is not None:
            closed = self._intent_closure_bitset(current)
            if closed not in seen:
                seen.add(closed)

                extent_bits = self._compute_extent(closed)
                support = self._support(extent_bits)

                # Titanic pruning: only keep if support >= min_support
                if self.min_support and support >= self.min_support:
                    concepts.append((extent_bits, closed))

            current = self._next_closure_bitset(current)

        return concepts

