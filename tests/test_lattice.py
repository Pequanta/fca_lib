from fca.concept_lattice import ConceptLattice
import numpy as np

def test_lattice_operations():
    # Example relation: 3 objects, 3 attributes
    # Each row is an object, each column is an attribute
    relation_matrix = np.array([
        [1, 0, 1],  # object 1 has attr1 and attr3
        [1, 1, 0],  # object 2 has attr1 and attr2
        [0, 1, 1],  # object 3 has attr2 and attr3
    ])
    # Encode each object's attributes as a bitset integer
    ext_int = np.array([int("".join(map(str, row)), 2) for row in relation_matrix])
    # Encode each attribute's objects as a bitset integer'
    int_ext = np.array([int("".join(map(str, relation_matrix[:, col])), 2) for col in range(relation_matrix.shape[1])])
    objects = np.array(['1', '2', '3'])
    attributes = np.array(['a', 'b', 'c'])

    lattice = ConceptLattice(ext_int, int_ext, objects, attributes)
    concepts = lattice.all_concepts()
    assert isinstance(concepts, list)
    assert len(concepts) > 0
    assert len(concepts) <= 2**3  # 2^3 possible concepts with 3 attributes

    # test with min_support
    lattice_min_sup = ConceptLattice(ext_int, int_ext, objects, attributes, min_support=0.2)
    concepts = lattice_min_sup.all_concepts()

    for extent, intent in concepts:
        assert isinstance(extent, int) or isinstance(extent, np.integer)
        assert isinstance(intent, int) or isinstance(intent, np.integer)

if __name__ == "__main__":
    test_lattice_operations()
    print("All tests passed.")