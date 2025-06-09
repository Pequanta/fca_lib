from typing import Set
from context import FormalContext
from utils.bitset import union
class FormalConcept:
    def __init__(self):
        self.extent = Set() 
        self.intent = Set()
        self.concept_base = 0 #it will indicate wheter the concept is formed based on attribute or object with object being assigned 0
        self.set_bit = 0 # it will hold the attributes common to concept's extent or objects common to concept's intent
    def extend_concept_from_object(self, context: FormalContext):
        """

        """
        self.set_bit = union(self.set_bit, 1 << context.attr)

    def extend_concept_from_attribute(self, context: FormalContext):
        pass
    
