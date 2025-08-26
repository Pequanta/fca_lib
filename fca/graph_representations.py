import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from fca.utils.utils import count_ones

matplotlib.use("TkAgg")
class BipartiteGraph:
    def __init__(self, relation_encoded: np.ndarray, objects: np.ndarray, attributes: np.ndarray) -> None:
        """
            Args:
                relation_encoded: contains bitset integers representing whether the relation exists between objects and attributes in 
                                  their binary form
                objects: a numpy array containing names of objects

                attributes: a numpy array containing names of attributes

            return:
                it returns a bipartite graph having objects on one side and attributes on the other and having edges between them


        """
        
        self.b_graph = nx.Graph()
        self.relation_encoded = relation_encoded
        self.objects = objects
        self.attributes = attributes
    def generate_graph(self):

        #adding objects and attributes as nodes
        self.b_graph.add_nodes_from(self.objects, bipartite=0)

        self.b_graph.add_nodes_from(self.attributes, bipartite=1)

        attr_size = len(self.attributes)
        #The following for loop extracts an edge existing between an object if index i with jth attribute
        for i in range(len(self.relation_encoded)):
            j = attr_size
            a = 0
            while j > 0:
                # << operator is used to check the attribute at a^th attribute 
                # the & operator returns 0 if the bitwise and operator is 0

                if (self.relation_encoded[i] & (1 << j)) != 0:
                    self.b_graph.add_edge(self.objects[i], self.attributes[a])
                a += 1
                j -= 1
        return self.b_graph
    def plot_graph(self):
        print(self.b_graph)
        pos = nx.bipartite_layout(self.b_graph, self.objects)

        plt.figure(figsize=(8, 5))
        nx.draw(
                    self.b_graph,
                    pos,
                    with_labels=True,
                    node_color=["skyblue" if n in self.objects else "lightgreen" for n in self.b_graph.nodes()],
                    node_size=4000
                    )
        plt.title("Bipartite Graph")
        plt.show()
class RandomGraph:
    def __init__(self, concepts, attributes, objects):
        self.concepts = concepts
        self.attributes = attributes
        self.objects = objects
    def bitset_to_attrset(self, bits: int) -> set[str]:
        return {self.attributes[i] for i in range(len(self.attributes)) if bits & (1 << i)}

    def bitset_to_objset(self, bits: int) -> set[str]:
        return {self.objects[i] for i in range(len(self.objects)) if bits & (1 << i)}
    def build_lattice_graph(self):
        """
        Builds a concept lattice (Hasse diagram) as a NetworkX graph.
        
        :param concepts: List of (intent_bitset, extent_bitset) pairs
        :param attribute_order: Ordered list of attributes (used for sorting)
        :return: (graph, pos, labels) tuple
        """
        concepts = [(obj, att) for obj, att in self.concepts]  # Keep as bitsets
        print(concepts)
        # Sort by intent size (ascending)
        concepts.sort(key=lambda c: count_ones(c[1]))

        G = nx.DiGraph()
        node_map = {}

        # Add nodes
        for obj, att in concepts:
            node_id = att
            node_map[node_id] = (obj, att)
            G.add_node(node_id)
        # Build covering relations using bitwise subset logic
        for i, (obj1, att1) in enumerate(concepts):
            for j in range(i + 1, len(concepts)):
                obj2, att2 = concepts[j]

                # Check if att1 ⊂ att2
                if att1 & att2 == att1 and att1 != att2:
                    # Check cover condition: no intermediate concept between att1 and att2
                    is_cover = True
                    for k in range(i + 1, j):
                        _, att3 = concepts[k]
                        if att1 & att3 == att1 and att3 & att2 == att3:
                            is_cover = False
                            break
                    if is_cover:
                        G.add_edge(att2, att1)  # top-down


        # Compute layout positions (layered by intent size)
        pos = {}
        layers = {}

        # Group nodes by layer (based on length)
        for node in G.nodes:
            layer = count_ones(node)
            layers.setdefault(layer, []).append(node)

        # Assigning positions
        for layer, nodes in layers.items():
            n = len(nodes)
            offset = -(n - 1)  # This centers the layer horizontally
            for i, node in enumerate(nodes):
                x = (i * 2) + offset  # Centering
                y = -layer
                pos[node] = (x, y)
        # Building labels
        labels = {
            node: f"I: {{{', '.join(sorted(self.bitset_to_attrset(node_map[node][1])))}}}\nE: {{{', '.join(list(map(lambda x: str(x), sorted(self.bitset_to_objset(node_map[node][0])))))}}}"
            for node in G.nodes
        }

        return G, pos, labels

    def plot_graph(self):
        G, pos, labels = self.build_lattice_graph()
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            labels=labels,
            with_labels=True,
            node_color="#add8e6",
            node_size=1800,
            font_size=10,
            font_family="monospace",
            font_weight="bold",
            edge_color="gray",
        )
        plt.title("Concept Lattice (Hasse Diagram)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()




#if __name__=="__main__":
#    inst = BipartiteGraph()
