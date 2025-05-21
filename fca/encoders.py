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
        print(data.columns)
        attributes_ = np.asarray(data.columns[2:])
        objects_ = data[data.columns[1]].to_numpy()
        data = data[attributes_]
        relation_to_numpy = data.to_numpy()
        relation_encoded = np.asarray([self.bitset_from_row(row) for row in relation_to_numpy])
        return [relation_encoded, objects_, attributes_]
    
    def numpy_encoder(self, data) -> np.ndarray:
        if type(data) != np.ndarray:
            print("Unsupported Type!")
            return np.ndarray([])
        
        relation_encoded = np.asarray([self.bitset_from_row(row) for row in data])
        return relation_encoded

                        
    def bitset_from_row(self, row):
        index = len(row) # the bit field length
        result = 0
        i = 0
        while index > 0:
            if row[i]:
                hold = 1 << index #to reach to the index that describes the relation between the object and the attribute of row[index]
                result |= hold
            index -= 1
            i += 1
        return result


