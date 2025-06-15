import pandas as pd
import numpy as np


class CsvReader:
    def __init__(self, file_path: str):
        #file path 
        self.file_path = file_path

    def pandas_dataframe(self):
        df = pd.read_csv(self.file_path)
