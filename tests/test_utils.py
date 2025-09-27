from fca.utils.utils import count_ones, gini_impurity_from_counts, gini_gain_over_baseline

def test_count_ones():
    assert count_ones(0b0) == 0
    assert count_ones(0b1) == 1
    assert count_ones(0b101010) == 3
    assert count_ones(0xFFFFFFFF) == 32

def test_gini_impurity_from_counts():
    assert gini_impurity_from_counts({0: 0, 1: 0}) == 0.0
    assert gini_impurity_from_counts({0: 5, 1: 5}) == 0.5
    assert gini_impurity_from_counts({0: 1, 1: 3}) == 1 - ((1/4)**2 + (3/4)**2)
    assert gini_impurity_from_counts({}) == 0.0

def test_gini_gain_over_baseline():
    baseline = {0: 5, 1: 5}
    extent = {0: 10, 1: 0}
    gain = gini_gain_over_baseline(extent, baseline)
    assert gain > 0
    assert gini_gain_over_baseline({0: 5, 1: 5}, {0: 5, 1: 5}) == 0.0

if __name__ == "__main__":
    test_count_ones()
    test_gini_impurity_from_counts()
    test_gini_gain_over_baseline()
    print("All utils tests passed.")