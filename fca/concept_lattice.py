from numpy import asarray, ndarray
from typing import List, Set
from utils.bitset import BitSetOperations


set_operations = BitSetOperations()

class ConceptLattice:
    def __init__(self, intent_extent: ndarray, extent_intent: ndarray, attr: ndarray, obj: ndarray):
        self.intent_extent = intent_extent
        self.extent_intent = extent_intent
        self.attr = attr
        self.obj = obj

    def generate_lattice(self):
        pass


    def intent_hierarchy(self):
        temp_hierarchy = {}
        

        for i in range(len(self.attr)):
            temp_hierarchy[self.attr[i]] = self.extent_union(self.intent_extent[i])

        return temp_hierarchy
    def extent_union(self, extents: int):
        #print(bin(extents))
        i = len(self.obj)
        extent_lst = []
    
        while i > 0:
            if ((1 << i) & extents) != 0:
                extent_lst.append(i)
            i -= 1

        
        intents_included = set_operations.union(extent_lst) #all intents that can be reached from the given intent
        print(bin(intents_included))
        result = [self.attr[i] for i in range(len(self.attr)) if (intents_included & (1 << i)) != 0]
        return result
    
class ConceptLatticeOperations:
    def __init__(self):
        pass

