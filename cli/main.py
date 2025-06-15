
from fca.encoders import pandas_encoder
from ucimlrepo import fetch_ucirepo

# fetch dataset
zoo = fetch_ucirepo(id=111)

# data (as pandas dataframes)
X = zoo.data.features
y = zoo.data.targets


print("Hello world")
class TestPandasConverter:
    def __init__(self):
        pass
class TestNumpyConverter:
    def __init__(self):
        pass
        




    
