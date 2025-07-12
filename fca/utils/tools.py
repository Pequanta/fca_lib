def count_ones(n: int) -> int:
    n = n & 0xFFFFFFFF  # ensure 32-bit representation
    n = n - ((n >> 1) & 0x55555555)
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
    n = (n + (n >> 4)) & 0x0F0F0F0F
    n = n + (n >> 8)
    n = n + (n >> 16)
    return n & 0x3F
