import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../fca')))

import numpy as np
import pandas as pd
from fca.encoders import Encoder
from fca.concept_lattice import ConceptLattice
from fca.graph_representations import RandomGraph
from fca.algorithms.iceberg_concept import IcebergConcept
from fca.qubo_formulation.qubo_formulations import QuboFormulation
from fca.qubo_formulation.classical_solutions import ClassicalSolutions
from fca.graph_representations import BipartiteGraph


encoder = Encoder()


df_temp = pd.read_csv("fca/assets/test_df.csv")
df_temp.drop(df_temp.columns[7:], axis=1, inplace=True)
df_temp = df_temp.iloc[:10]
print(df_temp.shape)
encoded_data, encoded_data_transposed, objects_, attributes_ = encoder.pandas_encoder(df_temp)

encoded_data, encoded_data_transposed, objects_, attributes_ = encoder.pandas_encoder(df_temp)

cont_test = list(map(lambda x: np.binary_repr(x, width=len(attributes_)), encoded_data))

concept_lattice = ConceptLattice(encoded_data, encoded_data_transposed, objects_,attributes_)
concepts = concept_lattice.all_concepts()
# print(concepts)
graph_ = RandomGraph(concepts, list(attributes_), list(objects_))
# graph_2 = BipartiteGraph(encoded_data, attributes_, objects_)
# graph_2.generate_graph()
# graph_2.plot_graph()

g_ , p_ , l_ = graph_.build_lattice_graph()
print(g_.nodes)
print(g_.nodes)
print(p_)

graph_.plot_graph()
# iceberg_concepts = IcebergConcept()
# print(iceberg_concepts.extract_iceberg_concepts(g_, 2))


# qubo_formulation = QuboFormulation()
# classical_solutions = ClassicalSolutions()
# print("Iceberg Concepts: ", IcebergConcept().extract_iceberg_concepts(concepts, 2)) # type: ignore
# print("QUBO Matrix: ") # type: ignore
# np.set_printoptions(precision=2, suppress=True)
# print(qubo_formulation.build_iceberg_qubo(concepts, 2)) # type: ignore
# print("QUBO Matrix Shape: ", qubo_formulation.build_iceberg_qubo(concepts, 2).shape) # type: ignore
# print("Classical Solutions: ") # type: ignore
# print(classical_solutions.solve_qubo_brute_force(qubo_formulation.build_iceberg_qubo(concepts, 2))) # type: ignore
# # graph_.plot_graph()


