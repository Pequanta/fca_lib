from models import BitSet
class BitSetOperations:
    """
        This class will handle initialization and different operations on binary relation sets.
        Set in this context is a class and its binary members are represented with BitSetegers whose binary
        representation will hold the relation between a given object and all of the attributes
    """

    def __init__(self):
        pass
    def __relation_exist__(self, attr_index: int , obj_attrs: BitSet) -> bool:
        return ((1 << attr_index) & 1) == 1

    def __subset_of__(self , obj_one_attrs: BitSet, obj_two_attrs: BitSet)-> bool:
        """
            checks if obj_one_attrs.set is a subset of obj_two_attrs.set
        """
        flag_exist = True #This will hold if the relations in the first object's set also exist in the second object's attributs
        bit_length = len(bin(obj_two_attrs.set)) - 2

        for i in range(bit_length):
            hold = 1 << i #This will hold which object-attribute relation to check

            if (obj_one_attrs.set & obj_two_attrs.set) & hold != 0:
                flag_exist = False
                break
        return (obj_one_attrs.set < obj_two_attrs.set) and flag_exist
    
    def __proper_subset_of__(self, obj_one_attrs: BitSet, obj_two_attrs: BitSet)-> bool:
        """
            checks if obj_one_attrs.set is a a proper subset of obj_two_attrs.set
        """
        return (obj_one_attrs.set == obj_two_attrs.set) or self.__subset_of__(obj_one_attrs.set, obj_two_attrs.set)

    def __superset_of(self, obj_one_attrs: BitSet, obj_two_attrs: BitSet)-> bool:
        pass

    def __equal_sets__(self, obj_one_attrs: BitSet, obj_two_attrs: BitSet) -> bool:
        """
            checks equality between obj_one_attrs.set and obj_two_attrs.set
        """
        return obj_one_attrs.set == obj_two_attrs.set
    

    def union(self, obj_one_attrs: BitSet, obj_two_attrs: BitSet) -> BitSet:
        return obj_one_attrs.set & obj_two_attrs.set

    def BitSetersection(self, obj_one_attrs: BitSet, obj_two_attrs: BitSet)-> BitSet:
        return obj_one_attrs.set | obj_two_attrs.set

    def difference(self, obj_one_attrs: BitSet, obj_two_attrs: BitSet)-> BitSet:
        return obj_one_attrs.set ^ obj_two_attrs.set

    def symmetric_difference(self, obj_one_attrs: BitSet, obj_two_attrs: BitSet)-> BitSet:
        max_length = max(len(bin(obj_one_attrs.set)), len(bin(obj_two_attrs.set)))
        bit_length = max_length - 2
        mask = (1 << bit_length) - 1
        sym_difference = (obj_one_attrs.set ^ obj_two_attrs.set) ^ mask 
        return sym_difference

    def complement(self, obj_attrs: BitSet, universal_set: BitSet)-> BitSet:
        return self.symmetric_difference(obj_attrs, universal_set)
        



