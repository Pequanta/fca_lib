import numpy as np
from fca.graph_representations import BipartiteGraph, RandomGraph
from fca.utils.utils import count_ones

def test_bipartite_graph_generation():
    # 2 objects, 3 attributes
    relation_encoded = np.array([0b101, 0b110])  # obj1: a,c; obj2: a,b
    objects = np.array(['obj1', 'obj2'])
    attributes = np.array(['a', 'b', 'c'])
    graph = BipartiteGraph(relation_encoded, objects, attributes)
    g = graph.generate_graph() 
    # Should have nodes less than or equal to 2^(number of attributes + number of objects)
    assert len(g.nodes) <= 2 ** 5
    # Should have edges for each relation
    assert g.number_of_edges() <= count_ones(relation_encoded[0]) + count_ones(relation_encoded[1])

def test_random_graph_bitset_conversion():
    concepts = [(0b11, 0b101), (0b10, 0b110)]
    attributes = ['a', 'b', 'c']
    objects = ['obj1', 'obj2']
    rg = RandomGraph(concepts, attributes, objects)
    attrset = rg.bitset_to_attrset(0b101)
    objset = rg.bitset_to_objset(0b11)
    assert attrset == {'a', 'c'}
    assert objset == {'obj1', 'obj2'}

if __name__ == "__main__":
    test_bipartite_graph_generation()
    test_random_graph_bitset_conversion()
    print("All tests passed.")