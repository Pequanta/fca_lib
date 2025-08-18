from numpy import ndarray, binary_repr
import pandas as pd
from typing import List, Tuple
from fca.encoders import Encoder
from fca.utils.bitset import BitSetOperations

bset_operations = BitSetOperations()
encoder = Encoder()


class PreprocessingJSM:
    def __init__(self, data: ndarray | pd.DataFrame):
        if type(data) == pd.DataFrame:
            ##Getting the target class
            self.goal_attr = list(data[data.columns[-1]])

            #The encoder class should only be used with the structural attributes. So the 
            #target class should be removed from the attributes list

            data.drop(columns=[data.columns[-1], data.columns[0]], axis=1, inplace=True)

            self.negative_examples = data.drop(labels=[index for index, _ in data.iterrows() if self.goal_attr[index] == False], inplace=True) # type: ignore
            self.positive_examples = data.drop(labels=[index for index, _ in data.iterrows() if self.goal_attr[index] == True], inplace=True) # type: ignore
            _, _ , self.objects, self.attributes = encoder.pandas_encoder(data) # type: ignore

            self.unknown_examples = data
            self.size = len(data.columns) - 1

        elif type(data) == ndarray:
            self.object_attribute_encoded , self.objects_, self.attributes_ = encoder.numpy_encoder(data)
        else:
            print("Not recognized type")


    def encode_data(self,  type: int):
        """
            Args:
                type: postive , negative or unknown

        """
        print("Check data: ", self.positive_examples)
        if type == 1:
             data_ = self.positive_examples
        elif type == 0:
             data_ = self.unknown_examples
        else:
             data_ = self.negative_examples

        object_attribute_encoded,attribute_object_encoded,objects_,attributes_ = encoder.pandas_encoder(data_)

        return object_attribute_encoded, attribute_object_encoded, objects_, attributes_    

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
        pos_context = list(map(lambda x: x ^ (1 << self.size), pos_context))

        learning_context.extend(pos_context)
        learning_context.extend(list(negative_context))

        return learning_context
    
    
