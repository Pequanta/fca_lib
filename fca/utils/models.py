from pydantic import BaseModel
from typing import List

class FormalContext(BaseModel):
    object: int
    attr: int

class FormalConcept(BaseModel):
    concept: List

class ConceptLattice(BaseModel):
    pass
