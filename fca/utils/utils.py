def count_ones(n: int) -> int:
    n = n & 0xFFFFFFFF  # ensure 32-bit representation
    n = n - ((n >> 1) & 0x55555555)
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
    n = (n + (n >> 4)) & 0x0F0F0F0F
    n = n + (n >> 8)
    n = n + (n >> 16)
    return n & 0x3F


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


