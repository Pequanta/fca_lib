import numpy as np
import pandas as pd
import os


class Encoder:
    def __init__(self):
        pass
    def pandas_encoder(self, data) -> np.ndarray | bool:
        if type(data) != pd.DataFrame:
            print("Unsupported Type!")
            return False

        relation_to_numpy = data.to_numpy()
        relation_encoded = np.asarray([self.bitset_from_row(row)] for row in relation_to_numpy)
        return relation_encoded
    
    def numpy_encoder(self, data) -> bool:
        if type(data) != np.ndarray:
            print("Unsupported Type!")
            return False
        
        relation_encoded = np.asarray([self.bitset_from_row(row)] for row in data)
        return relation_encoded
   
    def bitset_from_row(self, row):
        index = len(row) # the bit field length
        result = 0
        i = 0
        while index >= 0:
            if row[i]:
                hold = 1 << index #to reach to the index that describes the relation between the object and the attribute of row[index]
                result |= hold
                index -= 1
            i += 1
        return result
        