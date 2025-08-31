from numpy import ndarray, binary_repr
import pandas as pd
from typing import List, Tuple
from fca.encoders import Encoder
from fca.utils.bitset import BitSetOperations

bset_operations = BitSetOperations()
encoder = Encoder()


class PreprocessingJSM:
    def __init__(self, data_train: ndarray | pd.DataFrame,  goal_train: List[bool]):
        if type(data_train) == pd.DataFrame:
            ##Getting the target class
            self.goal_train = goal_train

            #The encoder class should only be used with the structural attributes. So the 
            #target class should be removed from the attributes list

            data_train.drop(columns=[data_train.columns[-1]], axis=1, inplace=True)

            self.object_attribute_encoded, self.attribute_object_encoded, self.objects, self.attributes = encoder.pandas_encoder(data_train) # type: ignore
            self.negative_examples = data_train.loc[[i for i, val in data_train.iterrows() if self.goal_train[i] == False]] #type: ignore   
            self.positive_examples = data_train.loc[[i for i, val in data_train.iterrows() if self.goal_train[i] == True]] #type: ignore

            self.size = len(data_train.columns) - 1

        elif type(data_train) == ndarray:
            self.object_attribute_encoded , self.objects_, self.attributes_ = encoder.numpy_encoder(data_train)
        else:
            print("Not recognized type")


    def encode_data(self,  type: int):
        """
            Args:
                type: postive , negative or unknown

        """
        if type == 1:
             data_ = self.positive_examples
        else:
             data_ = self.negative_examples
        object_attribute_encoded,attribute_object_encoded,objects_,attributes_ = encoder.pandas_encoder(data_) #type: ignore

        return object_attribute_encoded, attribute_object_encoded, objects_, attributes_

    @staticmethod
    def encode_test_data(data_test: pd.DataFrame):
        data = data_test.reset_index(drop=True)
        object_attribute_encoded, attribute_object_encoded, objects_, attributes_ = encoder.pandas_encoder(data) #type: ignore
        return object_attribute_encoded, attribute_object_encoded, objects_, attributes_

    def process_data(self) -> Tuple[List[int], List[int]]:
        """   
            Args:
                data (List[int]): The input data to be processed.
            
            Returns:
                Tuple[List[int]]: processed contexts
        """
        pos_context , _ , _ , _= self.encode_data(1)
        negative_context, _ , _, _ = self.encode_data(-1)
        return (list(pos_context), list(negative_context))
    def get_learning_data(self):
        """
            Create an instance of JSMMethodApplication with the provided contexts and objects/attributes.
            
            Args:
                pass
            Returns:
                learning_context (List[int]): A list of integers representing the learning context.
        """

        learning_context = []
        pos_context, negative_context = self.process_data()
        pos_context = list(map(lambda x: x ^ (1 << self.size), pos_context))

        learning_context.extend(pos_context)
        learning_context.extend(list(negative_context))

        return learning_context
    
    
