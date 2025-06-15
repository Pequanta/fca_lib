class ConceptLattice:
    def __init__(self):
        self.concept_lattice = dict()
    def add_element(self, element: str) -> bool:
        """
            Args:
                element: an object or attribute
        """
        #The check will be whether the object or attirbute is already in the lattice or not
        if element not in self.concept_lattice:
            self.concept_lattice[element] = []
            return True
        return False
    def remove_element(self, element: str) -> bool:
        if element not in self.concept_lattice:
            delete self.concept_lattice[element]
            return True
        return False
    ##The following method only serves for the debugging part

    def print_lattice(self):
        print("{")
        for element in self.concept_lattice:
            print(f"{element}: {self.concept_lattice[element]}")
        print("}")   
