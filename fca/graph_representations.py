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
class OrderedGraph:
    def __init__(self, intent_with_extent_length, nodes):
        
        self.graph = nx.Graph()
        self.extent_lengths = intent_with_extent_length
        self.graph.add_nodes_from(nodes)
    
    def create_an_edge(self):
        pass

    def pos_location(self, extent_lengths):
        """
            Args:
                extent_lengths: a dictionary containing intents : len(extents encompassed directly by the intent)
            Output:
                pos_dictionary: a dictionary containing pos: List[intents] indicating relative positions for the intents in the lattice graph
        """
        y = 0
        sorted_intents = sorted(extent_lengths.keys(), key = lambda x: extent_lengths[x])
        mid_length = max(extent_lengths.values()) // 2 
        pos_temp_dictionary = {}
        prev = None
        for i in range(len(sorted_intents)):
            pos_ = extent_lengths[sorted_intents[i]]
            if prev and pos_ != prev:
                y += 1
                pos_temp_dictionary[y] = [sorted_intents[i]]
            elif not prev:
                pos_temp_dictionary[0] = [sorted_intents[i]]
            else:
                pos_temp_dictionary[y].append(sorted_intents[i])
            prev = pos_

        pos_dictionary = {}
        prev = None
        for length in pos_temp_dictionary:
            x = 0
            for element in pos_temp_dictionary[length]:
                   pos_dictionary[element] = (x, y)
                   x += 1
            y += 1
        return pos_dictionary
    def generate_graph(self):
        pos = self.pos_location(self.extent_lengths)
        self.graph.add_nodes_from(self.extent_lengths)
        nx.draw(self.graph, pos, with_labels=True, node_size=1000, alpha=0.5)
        
        plt.show()
#if __name__=="__main__":
#    inst = BipartiteGraph()
