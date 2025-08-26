import pandas as pd
import numpy as np

from fca.encoders import Encoder

encoder = Encoder()

class FuzzyLogic:
    def __init__(self, data: pd.DataFrame, class_column: str, object_column: str):
        self.data = data
        self.type_ = pd.DataFrame if type(data) == pd.DataFrame else np.ndarray

        self.object_col = object_column
        self.class_col = data[class_column]
        self.data.drop(columns=[class_column], inplace=True)

        for i, value in self.class_col.items():
                self.class_col[i] = float(True) if value  == 'M' else float(False)

    def binarize_data_multiple(self, numeric_cols, class_col):
        """
            Convert pd.DataFrame into a list of binary attribute dicts per object.
            Args:
                numeric_cols (list): List of numeric columns to be binarized.
                class_col (str): Name of the class column.
            Returns (bin_rows, thresholds_dict, attribute_list)
        """
        thresholds = {}
        for col in numeric_cols:
            thresholds[col] = self.get_multiple_threshold(self.data[col].tolist(), self.class_col.tolist())
        
        bin_rows = []
        attributes = set()
        for _, row in self.data.iterrows():
            br = {}
            for col in self.data.columns:
                if col in numeric_cols:
                    for thr in thresholds[col]:
                        a1 = f"{col} <= {thr}"
                        a2 = f"{col} >= {thr}"
                        br[a1] = (row[col] <= thr)
                        br[a2] = (row[col] >= thr)
                        attributes.add(a1); attributes.add(a2)
                else:
                    # categorical or already discrete
                    a = f"{col} == {row[col]}"
                    br[a] = True
                    attributes.add(a)
            bin_rows.append(br)
        return bin_rows, thresholds, sorted(attributes)
    def binarize_data_single(self):
        """
            Convert pd.DataFrame into a single binary attribute dict per object.
            Args:
                numeric_cols (list): List of numeric columns to be binarized.
                class_col (str): Name of the class column.
            Returns (bin_row, thresholds_dict, attribute_list)
        """
        self.fuzzy_to_binary()

        object_attribute_encoded, attribute_object_encoded, objects_, attributes_ = encoder.pandas_encoder(self.data)

        return (object_attribute_encoded, attribute_object_encoded, objects_, attributes_)
    




    def get_multiple_threshold(self, values, targets):
        """Finding midpoints between adjacent values where target changes."""
        paired = sorted(zip(values, targets), key=lambda x: (float('inf') if x[0] is None else x[0], x[1]))
        thresholds = []
        for (v1, t1), (v2, t2) in zip(paired, paired[1:]):
            if v1 is None or v2 is None:
                continue
            if (v1 - v2 >= sum(values) // len(values)) and t1 != t2:
                thresholds.append((v1 + v2) / 2.0)
        # unique & sorted
        return sorted(set(thresholds))
    
    def fuzzy_to_binary(self):

        row_size = self.data.shape[0]
        for col in self.data.columns:
            median_ = np.median(self.data[col])
            if col == self.object_col:
                continue
    
            for i in range(row_size):
                self.data.loc[i, col] = float(True) if self.data.loc[i, col] >= median_ else float(False)



        