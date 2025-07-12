
from itertools import product
import numpy as np
from utils.tools import count_ones
class ClassicalSolutions:
    """
    A class to represent classical solutions for FCA problems.
    """

    def __init__(self):
        pass


    def solve_qubo_brute_force(self, Q):
        n = Q.shape[0]
        best_solution = None
        best_energy = float('inf')

        for x in product([0, 1], repeat=n):
            x_vec = np.array(x)
            energy = x_vec @ Q @ x_vec  # QUBO objective: xᵀ Q x
            if energy < best_energy:
                best_energy = energy
                best_solution = x

        return best_solution, best_energy
