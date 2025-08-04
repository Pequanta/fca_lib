import numpy as np
import pandas as pd
from typing import List
class Encoder:
    def __init__(self):
        pass
    
    def pandas_encoder(self, data) -> List[np.ndarray]:
        """
            Args:
                data: a pandas framework  
                    : It is assumed that the data will hold the object names in it's first column and the attributes 
                      in column names except for the first column
            result:
                The method will extract an object array, attribute array and a relation encoded with bitset representation 
                from the passed pandas framework
        """
        if type(data) != pd.DataFrame:
            print("Unsupported Type!")
            return [np.asarray([])]


        attributes_ = np.asarray(data.columns[1:])

        objects_ = data[data.columns[1]].to_numpy()
        data = data[attributes_]

     
        relation_to_numpy = data.to_numpy()
        object_attribute_encoded = np.asarray([self.bitset_from_row(row) for row in relation_to_numpy])
        return [object_attribute_encoded, objects_, attributes_]
    
    def numpy_encoder(self, data) -> List[np.ndarray]:
        if type(data) != np.ndarray:
            print("Unsupported Type!")
            return [np.ndarray([])]
        attributes_ = data[0][2:]
        objects_ = np.asarray([data[i][1] for i in range(1, len(data))])
        object_attribute_encoded = np.asarray([self.bitset_from_row(row[1:]) for row in data[1:]])
        return [object_attribute_encoded, objects_, attributes_]
                   
    def bitset_from_row(self, row):
        result = (1 << len(row)) - 1
        i = len(row) - 1
        while i > 0:
            if row[i] not in {'True', True, 'X', 'x', '1', 1}:
                result ^= (1 << i)
            i -= 1
        return result


