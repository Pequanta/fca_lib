def count_ones(num: int) -> int:
    hold = 1
    count = 0
    while num > 0:
        count += hold & num
        num >>= 1
    return count