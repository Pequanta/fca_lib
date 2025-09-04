import os, sys
from scripts.examples.jsm_method.jsm_method import JSMMethodApplication
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../fca')))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
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



"""
    Concept lattice generation for small dataset
"""



df_small = pd.read_csv("fca/assets/test_df.csv")

def generate_concept_lattice(df):
    """
    Generates and plots the concept lattice for a given DataFrame.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        None
    """
    ext_int, int_ext, obj, attr = encoder.pandas_encoder(df) #type: ignore
    concept_lattice = ConceptLattice(ext_int, int_ext, obj, attr)
    concepts = concept_lattice.all_concepts()
    graph_ = RandomGraph(concepts, list(attr), list(obj))
    graph_.build_lattice_graph()
    graph_.plot_graph()







"""
    Benchmarking
"""
df_temp = pd.read_csv("fca/assets/breast-cancer.csv")

fuzzy_ = FuzzyLogic(df_temp, 'diagnosis', 'id')
fuzzy_.binarize_data_single()
fuzzy_.data['diagnosis'] = fuzzy_.class_col




def decision_tree_classifier(X_train, y_train, X_test):
    """
    Trains a Decision Tree classifier and predicts labels for the test set.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Predicted labels for the test set.
    """
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    return dt.predict(X_test)


def jsm_classifier(X_train, y_train, X_test, preprocessing_class, jsm_model_class, simulation_type="classical"):
    """
    Trains a JSM-based classifier and predicts labels for the test set.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        preprocessing_class (class): Preprocessing class for JSM.
        jsm_model_class (class): JSM classifier class.
        simulation_type (str, optional): Solution type ("classical" or "quantum"). Defaults to "classical".

    Returns:
        list[bool]: List of predicted labels for the test set.
    """
    goal_train = y_train.to_list()

    data_= preprocessing_class(X_train, goal_train)
    objects = data_.objects
    attributes = data_.attributes

    undetermined_context, _, _, _  = data_.encode_test_data(X_test)

    jsm_method = jsm_model_class(
            goal_attr=goal_train,
            objects_=list(objects),
            attributes_=list(attributes),
            data_preprocessing=data_,
            solution_type=simulation_type
        ) 
    
    jsm_method.train()
    result = [jsm_method.classify_(undetermined_context[i], i) == 1 for i in range(len(undetermined_context))]
    return result


def benchmark_jsm_vs_classical(df, numeric_cols, class_col, jsm_model_class, preprocessing_class, test_size=0.3, random_state=42):
    """
    Benchmark JSM-based classification against a Decision Tree classifier.

    Args:
        df (pd.DataFrame): Dataset with features and class column.
        numeric_cols (list): List of numeric feature names.
        class_col (str): Target column name.
        jsm_model_class (class): JSM classifier class.
        preprocessing_class (class): Preprocessing class for JSM.
        test_size (float, optional): Fraction of data to use as test set. Defaults to 0.3.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        dict: Accuracy scores {"decision_tree": str, "jsm_classical": str, "jsm_dirac": str}
    """

    # Split dataset
    df['diagnosis'] = fuzzy_.class_col.astype(bool)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['diagnosis'])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = train_df[numeric_cols]
    y_train = train_df[class_col]
    X_test = test_df.drop(columns=[class_col])
    y_test = test_df[class_col]

    #classification with decision tree
    y_pred_dt = decision_tree_classifier(X_train, y_train, X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)


    #classification with JSM with classical annealer
    y_pred_jsm_classical = jsm_classifier(X_train, y_train, X_test, jsm_model_class=jsm_model_class, preprocessing_class=preprocessing_class, simulation_type="classical")
    acc_jsm_classical = accuracy_score(y_test, y_pred_jsm_classical)


    #classification with JSM with dirac-3
    y_pred_jsm_dirac = jsm_classifier(X_train, y_train, X_test, jsm_model_class=jsm_model_class, preprocessing_class=preprocessing_class, simulation_type="quantum")
    acc_jsm_dirac = accuracy_score(y_test, y_pred_jsm_dirac)
    
    return {"decision_tree": f"{acc_dt:.3f}", "jsm_classical": f"{acc_jsm_classical:.3f}", "jsm_dirac": f"{acc_jsm_dirac:.3f}"}




if __name__ == "__main__":
    print(benchmark_jsm_vs_classical(df_temp, [col for col in df_temp.columns if col != 'diagnosis'], 'diagnosis', JSMMethodApplication, PreprocessingJSM)) # type: ignore
    #generate_concept_lattice(df_small)