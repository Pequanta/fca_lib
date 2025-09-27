def count_ones(n: int) -> int:
    return n.bit_count()  # enough to hold 0–64


# tools for iceberg logic 
# Information gain

def gini_impurity_from_counts(counts):
    total = sum(list(counts.values()))
    if total == 0:
        return 0
    probs = [counts[count] / total for count in counts]
    return 1 - sum(p ** 2 for p in probs)

def gini_gain_over_baseline(extent_counts, baseline_counts):
    base = gini_impurity_from_counts(baseline_counts)
    after = gini_impurity_from_counts(extent_counts)
    return max(0.0, base - after)


