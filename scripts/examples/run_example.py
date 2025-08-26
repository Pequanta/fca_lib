import os, sys

from scripts.examples.jsm_method.jsm_method import JSMMethodApplication
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
from jsm_method.data_preprocessing import PreprocessingJSM
from fca.utils.fuzzy_logic import FuzzyLogic
from typing import List, Tuple
from collections import Counter
encoder = Encoder()

df_temp = pd.read_csv("fca/assets/breast-cancer.csv")

fuzzy_ = FuzzyLogic(df_temp, 'diagnosis', 'id')


fuzzy_.binarize_data_single()



# bin_rows, thresholds, attributes = data_.binarize_data(num_columns, class_column)




# print(len(bin_rows), len(thresholds), len(attributes))
# encoded_data, encoded_data_transposed, objects_, attributes_ = fuzzy_.binarize_data_single()
# concept_lattice = ConceptLattice(encoded_data, encoded_data_transposed, objects_,attributes_)


# concepts = concept_lattice.all_concepts()
# print(len(concepts))
# graph_ = RandomGraph(concepts, list(attributes_), list(objects_))
# # # graph_2 = BipartiteGraph(encoded_data, attributes_, objects_)
# # # graph_2.generate_graph()
# # # graph_2.plot_graph()

# g_ , p_ , l_ = graph_.build_lattice_graph()
# print(g_.nodes)
# print(g_.nodes)
# print(p_)

# graph_.plot_graph()
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



"""
    Experimental code for JSM Method
"""
# get_data = df_temp['diagnosis'].to_list()
# goal_attr = [True  if get_data[i] == 'M' else False for i in range(len(get_data))]


# data_ = PreprocessingJSM(fuzzy_.data)
# goal_attr = goal_attr
# objects = data_.objects
# attributes = data_.attributes


# # print(unknown_data)

# # # print(object_attr_encoded, pos_context, negativate_context, goal_attr, objects_, attributes_)

# jsm_method = JSMMethodApplication(
#         goal_attr=goal_attr,
#         objects_=list(objects),
#         attributes_=list(attributes),
#         data_preprocessing=data_
#     ) 

# for i in range(len(undetermined_context)):
#     print(jsm_method.classify_(undetermined_context[i]))

# result = [jsm_method.classify_(undetermined_context[i]) for i in range(len(undetermined_context))  ]
# print(result)


# df_ = pd.read_csv("fca/assets/test_df2.csv")
# df_.head()

# data_ = PreprocessingJSM(df_)
# goal_attr = data_.goal_attr
# objects = data_.objects
# attributes = data_.attributes
# _, _, undetermined_context = data_.process_data()

# # print(unknown_data)

# # # print(object_attr_encoded, pos_context, negativate_context, goal_attr, objects_, attributes_)

# jsm_method = JSMMethodApplication(
#         goal_attr=goal_attr,
#         objects_=list(objects),
#         attributes_=list(attributes),
#         data_preprocessing=data_
#     ) 


# result = [jsm_method.classify_(undetermined_context[i]) for i in range(len(undetermined_context))]
# print(result)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def benchmark_jsm_vs_classical(df, numeric_cols, class_col, jsm_model_class, preprocessing_class, test_size=0.3, random_state=42):
    """
    Benchmark JSM-based classification against a Decision Tree classifier.
    
    Args:
        df (pd.DataFrame): Dataset with features and class column.
        numeric_cols (list): List of numeric feature names.
        class_col (str): Target column name.
        jsm_model_class: Your JSMMethodApplication class (already initialized with preprocessing).
        test_size (float): Fraction of data to use as test set.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Accuracy scores {"jsm": float, "decision_tree": float}
    """

    # Split dataset
    df['diagnosis'] = fuzzy_.class_col.astype(bool)
    print(df.head(20))
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['diagnosis'])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # ----- Classical baseline: Decision Tree -----
    X_train = train_df[numeric_cols]
    y_train = train_df[class_col]
    X_test = test_df.drop(columns=[class_col])
    y_test = test_df[class_col]


    dt = DecisionTreeClassifier(random_state=random_state)

    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    print("Y_train: ", Counter(y_train.to_list()))
    print("Y_test: ", Counter(y_test.to_list()))


    goal_train = y_train.to_list()
    goal_test = y_test.to_list()

    data_= preprocessing_class(X_train, X_test, goal_train, goal_test)
    objects = data_.objects
    attributes = data_.attributes

    # print("####")

    # print(data_.negative_examples.shape)
    # print(data_.positive_examples.shape)
    # print(data_.unknown_examples.shape)

    _, _, undetermined_context  = data_.process_data()

    # print("####")


    # # print(object_attr_encoded, pos_context, negativate_context, goal_attr, objects_, attributes_)

    jsm_method = jsm_model_class(
            goal_attr=goal_train,
            objects_=list(objects),
            attributes_=list(attributes),
            data_preprocessing=data_
        ) 
    unknown_context, _, _, _  = encoder.pandas_encoder(X_test) # type: ignore

    result = []
    
    result = [jsm_method.classify_(unknown_context[i]) for i in range(len(X_test))]

    acc_jsm = accuracy_score(list(y_test), result)
    return {"jsm": acc_jsm, "decision_tree": acc_dt}

fuzzy_.data['diagnosis'] = fuzzy_.class_col

print(benchmark_jsm_vs_classical(df_temp, [col for col in df_temp.columns if col != 'diagnosis'], 'diagnosis', JSMMethodApplication, PreprocessingJSM))