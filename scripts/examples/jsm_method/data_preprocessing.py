from numpy import ndarray, binary_repr
from pandas import DataFrame
from typing import List, Tuple
from fca.encoders import Encoder
from fca.utils.bitset import BitSetOperations

bset_operations = BitSetOperations()
encoder = Encoder()


class PreprocessingJSM:
    def __init__(self, data: ndarray | DataFrame):
        if type(data) == DataFrame:
            ##Getting the target class
            self.goal_attr = list(data[data.columns[-1]])

            #The encoder class should only be used with the structural attributes. So the 
            #target class should be removed from the attributes list
            data.drop(data.columns[-1], axis=1, inplace=True)

            #The length of the attributes can be calculated by taking the whole columns 
            #and subtracting one to exclude  the 'object_name' column
            self.num_attrs = len(data.columns) - 1
            self.object_attribute_encoded,\
            self.objects_,\
            self.attributes_ = encoder.pandas_encoder(data)
            print(*list(map(lambda x: binary_repr(x, width=4), (self.object_attribute_encoded))), sep="\n")

        elif type(data) == ndarray:
            self.object_attribute_encoded , self.objects_, self.attributes_ = encoder.numpy_encoder(data)
        else:
            print("Not recognized type")

    

    def process_data(self) -> Tuple[List[int], List[int], List[int]]:
            """
            
            Args:
                data (List[int]): The input data to be processed.
            
            Returns:
                Tuple[List[int]]: processed contexts
            """
            pos_context = []
            negative_context = []
            undetermined_context = []
            for i in range(len(self.goal_attr)):
                relation_ = self.object_attribute_encoded[i]
                if self.goal_attr[i] == True:
                     pos_context.append(relation_)
                elif self.goal_attr[i] == False:
                     negative_context.append(relation_)
                else:
                     undetermined_context.append(relation_)

            return (pos_context, negative_context, undetermined_context)

    def get_learning_data(self):
        """
        Create an instance of JSMMethodApplication with the provided contexts and objects/attributes.
        
        Args:
            pass
        Returns:
            learning_context (List[int]): A list of integers representing the learning context.
        """

        learning_context = []

        pos_context, negative_context, _  = self.process_data()

        pos_context = list(map(lambda x: x ^ (1 << self.num_attrs), pos_context))

        learning_context.extend(pos_context)
        learning_context.extend(list(negative_context))

        return learning_context
    
    
