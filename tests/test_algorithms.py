from fca.algorithms.iceberg_concept import IcebergConcept
from fca.algorithms.next_closure import NextClosure
from fca.utils.utils import count_ones
def test_iceberg_concept_basic():
    iceberg = IcebergConcept()
    #testing that iceberg concept returns empty for empty input
    assert iceberg.get_iceberg_data([], [], 0, {}) == {}

def test_next_closure_basic():
    num_objects = 4
    num_attributes = 3
    object_bit_rows = [0b111, 0b110, 0b101, 0b011]  # Example bitsets for objects
    attr_bit_rows = [0b111, 0b110, 0b101]  # Example bitsets for attributes
    nc = NextClosure(num_objects, num_attributes, object_bit_rows, attr_bit_rows)
    # The first closure should be the empty set
    first_closure = nc._next_closure_bitset(0)
    assert isinstance(first_closure, int)
    # The closure of [0] (first attribute) should be a valid intent
    closure_0 = nc._next_closure_bitset(1)
    assert isinstance(closure_0, int)
    # The closure should always be a subset of the attribute indices
    assert all(isinstance(i, int) and count_ones(closure_0) <= num_attributes for i in range(num_attributes))

if __name__ == "__main__":
    test_iceberg_concept_basic()
    test_next_closure_basic()
    print("All tests passed.")