import numpy as np
from typing import List, Set
from utils.bitset import BitSetOperations


set_operations = BitSetOperations()

class ConceptLattice:
    def __init__(self, intent_extent: np.ndarray, extent_intent: np.ndarray, attr: np.ndarray, obj: np.ndarray):
        self.intent_extent = intent_extent
        self.extent_intent = extent_intent
        self.attr = attr
        self.obj = obj
        # print("Intents: ", self.intent_extent)
        # print("Extents: ", extent_intent)

        # print("{")
        # for i in range(len(self.attr)):
        #     print(f"{self.attr[i]}: {np.binary_repr(self.intent_extent[i], width=5)}")


    def extent_derivation_operator(self, attr_set: List[int]):
        """
            args:
                a List of integers representing the indexes of the attributes in question
            Returns:
                a list of objects derived from passed attributes
        """

        common_objects = set_operations.intersection(attr_set)
        result = []

        index = 0 #keeps track of the index in the list
        left_index = len(self.obj) #keeps track of the index in the binary representation of the integer

        while left_index:
            if ((1 << left_index) & common_objects) != 0:
                result.append(self.obj[index])
            left_index -= 1
            index += 1
        print(result)
        return result 


        

    def intent_derivation_operator(self, obj_set: List[int]):
        """
            args:
                an integer representing the index of the attribute in question
            Returns:
                a list of attributes dervied from passed objects
        """

        common_attributes = set_operations.intersection(obj_set) #an integer whose binar representation holds
                                                                 #the indexes of objects common to the attributes
        result = []

        index = 0 #keeps track of the index in the list
        left_index = len(self.attr) #keeps track of the index in the binary representation of the integer
        while left_index:
            if ((1 << left_index) & common_attributes) != 0:
                result.append(index)
            left_index -= 1
            index += 1
        
        print(result)
        return result

    def generate_lattice(self):
        pass


    def intent_hierarchy(self):
        temp_hierarchy = {}
        

        for i in range(len(self.attr)):
            temp_hierarchy[self.attr[i]] = self.extent_union(i , self.intent_extent[i])
        print(temp_hierarchy)
        return temp_hierarchy
    def extent_union(self, intent: int, extents: int):
        """
            Args:
                extents: binary representation of the extents held by the requested intent
            result:
                result: an array containing reachable intents resulted from the union of intents held by the extents passed in the arg.
        """
        #print(bin(extents))
        i = len(self.obj) - 1
        ##The following list will contain the extents extracted from the binary representations passed through the argument
        extent_lst = []
        index = 0 
        while i >= 0:
            if ((1 << i) & extents) != 0:
                extent_lst.append(index)
            i -= 1
            index += 1
            

        intent_lst = [self.extent_intent[index] for index in extent_lst]

        intents_included = set_operations.union(intent_lst) #all intents that can be reached from the given intent

        i = len(self.attr) - 1
        result = []
        index = 0
        while i >= 0:
            #To be included in the intent's reach , the tables index of (row, column) = (intent, index) = True 
            # and index should be different from the required intent's index 
            if ((1 << i) & intents_included != 0) and index != intent:
                result.append(self.attr[index])
            index += 1
            i -= 1


        return result
    
    
class ConceptLatticeOperations:
    def __init__(self):
        pass

