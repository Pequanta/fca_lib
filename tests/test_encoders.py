import numpy as np
import pandas as pd
from fca.encoders import Encoder

def test_bitset_from_row():
    encoder = Encoder()
    row = [1, 0, 'X', 'x', 'True', False]
    bitset = encoder.bitset_from_row(row)
    # Should set bits for 1, 'X', 'x', 'True'
    # Positions 0, 2, 3, 4 should be set
    expected = (1 << 6) - 1
    expected ^= (1 << 1)  # 0 at index 1
    expected ^= (1 << 5)  # False at index 5
    assert bitset == expected

def test_pandas_encoder():
    encoder = Encoder()
    df = pd.DataFrame({
        'obj': ['o1', 'o2', 'o3'],
        'a': [1, 0, 1],
        'b': [0, 1, 1],
        'c': ['X', 'x', 0]
    })
    result = encoder.pandas_encoder(df)
    assert result is not None
    object_attribute_encoded, attribute_object_encoded, objects_, attributes_ = result
    assert len(object_attribute_encoded) == 3
    assert len(attribute_object_encoded) == 3
    assert list(objects_) == ['o1', 'o2', 'o3']
    assert list(attributes_) == ['a', 'b', 'c']

def test_numpy_encoder():
    encoder = Encoder()
    # Simulate a numpy array with shape (4, 5)
    # First row: header, next rows: data
    data = np.array([
        ['idx', 'obj', 'a', 'b', 'c'],
        [0, 'o1', 1, 0, 'X'],
        [1, 'o2', 0, 1, 'x'],
        [2, 'o3', 1, 1, 0]
    ], dtype=object)
    result = encoder.numpy_encoder(data)
    assert isinstance(result, list)
    assert len(result) == 4
    relation_encoded, inverse_relation_encoded, objects_, attributes_ = result
    assert len(relation_encoded) == 3
    assert len(objects_) == 3
    assert len(attributes_) == 3

if __name__ == "__main__":
    test_bitset_from_row()
    test_pandas_encoder()
    test_numpy_encoder()
    print("All tests passed.")