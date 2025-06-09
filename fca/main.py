# #from lattice import ConceptLattice
# from encoders import Encoder
#
#
#
# from ucimlrepo import fetch_ucirepo
#
# # fetch dataset
#
# zoo = fetch_ucirepo(id=111)
#
# # data (as pandas dataframes)
# X = zoo.data.features
# X.drop(columns=["legs"], inplace=True)
# y = zoo.data.targets
#
#
# encoder_  = Encoder()
# hold_res = encoder_.pandas_encoder(X)
#
# print(hold_res, len(hold_res))
# #print(*hold_res, sep="\n")


from numpy import sort
import pandas as pd
from graph_representations import BipartiteGraph, OrderedGraph
from encoders import Encoder


encoder = Encoder()

df = pd.read_csv("assets/test_df.csv")
encoded_data, encoded_data_transposed, objects_, attributes_ = encoder.pandas_encoder(df)
graph = BipartiteGraph(encoded_data, objects_, attributes_)
graph_2 = BipartiteGraph(encoded_data_transposed, objects_, attributes_)

graph_generated = graph.generate_graph()

#this will hold the intensions with their connected extensions
cont_temp = {intent: list(graph_generated.neighbors(intent)) for intent, d in graph_generated.nodes(data=True) if d["bipartite"] == 1}
hold_lsts = []

#the following dictionary will held the length values for intents
print(cont_temp)
       
extent_lengths = {intent: len(extents) for intent, extents in cont_temp.items()}
#print("location: ",  pos_location(extent_lengths))

graph_ = OrderedGraph(extent_lengths, attributes_)
#print("relationship: " , cont_temp)
#graph.plot_graph()
graph_.generate_graph()
#graph_2.plot_graph()
