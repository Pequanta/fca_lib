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
            
        print(np.binary_repr(extents, width = len(self.obj)), extent_lst)
       

        intent_lst = [self.extent_intent[index] for index in extent_lst]

        print(intent_lst) 
        intents_included = set_operations.union(intent_lst) #all intents that can be reached from the given intent

        print("intents: " , np.binary_repr(intents_included, width=len(self.attr)))
        i = len(self.attr) - 1
        result = []
        index = 0
        while i >= 0:
            if ((1 << i) & intents_included != 0) and index != intent:
                result.append(self.attr[index])
            index += 1
            i -= 1


        return result
    
class ConceptLatticeOperations:
    def __init__(self):
        pass

