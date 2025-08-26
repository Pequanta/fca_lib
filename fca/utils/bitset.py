from typing import List
class BitSetOperations:
    """
        This class will handle initialization and different operations on binary relation sets.
        Set in this context is a class and its binary members are represented with integers whose binary
        representation will hold the relation between a given object and all of the attributes
    """

    def __init__(self):
        pass
    def __relation_exist__(self, attr_index: int , obj_attrs: int) -> bool:
        return ((1 << attr_index) & obj_attrs) == 1

    def __subset_of__(self , obj_one_attrs: int, obj_two_attrs: int)-> bool:
        """
            checks if obj_one_attrs.set is a subset of obj_two_attrs.set
            (i.e., all bits set in obj_one_attrs are also set in obj_two_attrs)
        """
        return (obj_one_attrs & obj_two_attrs) == obj_one_attrs
    
    def __proper_subset_of__(self, obj_one_attrs: int, obj_two_attrs: int)-> bool:
        """
            checks if obj_one_attrs is a a proper subset of obj_two_attrs
        """
        return (obj_one_attrs == obj_two_attrs) or self.__subset_of__(obj_one_attrs, obj_two_attrs)

    def __superset_of(self, obj_one_attrs: int, obj_two_attrs: int)-> bool:
        pass

    def __equal_sets__(self, obj_one_attrs: int, obj_two_attrs: int) -> bool:
        """
            checks equality between obj_one_attrs.set and obj_two_attrs.set
        """
        return obj_one_attrs == obj_two_attrs
    

    def union(self, obj_attrs: List[int]) -> int:
        if len(obj_attrs) == 0:
            return 0
        result = obj_attrs[0]
        for i in range(1, len(obj_attrs)):
            result |= obj_attrs[i] 
        return result

    def intersection(self, obj_attrs: List[int])-> int:
        result = obj_attrs[0]
        for i in range(1, len(obj_attrs)):
            result &= obj_attrs[i] 
        return result
    def difference(self, obj_attrs: List[int])-> int:
        result = obj_attrs[0]
        for i in range(1, len(obj_attrs)):
            result ^= obj_attrs[i] 
        return result

    def symmetric_difference(self, obj_one_attrs: int, obj_two_attrs: int)-> int:
        
        max_length = max(len(bin(obj_one_attrs)), len(bin(obj_two_attrs)))
        bit_length = max_length - 2
        mask = (1 << bit_length) - 1
        sym_difference = (obj_one_attrs ^ obj_two_attrs) ^ mask 
        return sym_difference

    def complement(self, obj_attrs: int, universal_set: int)-> int:
        return self.symmetric_difference(obj_attrs, universal_set)
        

    def check_valid_galios_connect(self):
        pass
    def check_fixpoints(A, B):
        pass

