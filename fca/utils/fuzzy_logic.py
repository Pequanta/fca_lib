from pandas import DataFrame
from numpy import ndarray
class FuzzyLogic:
    def __init__(self, data: DataFrame | ndarray):
        self.data = data
        self.type_ = DataFrame if type(data) == DataFrame else ndarray

    def fuzzy_simple(self, threshold: float = 0.5) -> ndarray | DataFrame:
        if type(self) == DataFrame:
            

    def fuzzy_logic(self):
        pass