import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 


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

        self.relation_encoded = relation_encoded
        self.objects = objects
        self.attributes = attributes
    def generate_graph(self):
        self.b_graph = nx.Graph()

        #adding objects and attributes as nodes
        self.b_graph.add_nodes_from(self.objects, bipartite=0)

        self.b_graph.add_nodes_from(self.attributes, bipartite=1)

        attr_size = len(self.attributes)
        #The following for loop extracts an edge existing between an object if index i with jth attribute

        for i in range(len(self.relation_encoded)):
            j = attr_size
            while j > 0:
                # << operator is used to check the attribute at jth attribute 
                # the & operator returns 0 if the bitwise and operator is 0

                if (self.relation_encoded[i] & (1 << j)) != 0:
                    self.b_graph.add_edge(self.objects[i], self.attributes[j - 1])
                j -= 1
 
        return self.b_graph
    def plot_graph(self):
        pos = nx.bipartite_layout(self.b_graph, self.objects)

        plt.figure(figsize=(8, 5))
        nx.draw(
                    self.b_graph,
                    pos,
                    with_labels=True,
                    node_color=["skyblue" if n in self.objects else "lightgreen" for n in self.b_graph.nodes()],
                    node_size=1000
                )
        plt.title("Bipartite Graph")
        plt.show()

