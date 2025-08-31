
from itertools import product
import numpy as np
from fca.utils.utils import count_ones
import random
import math
class ClassicalSolutions:
    """
    A class to represent classical solutions for FCA problems.
    """

    def __init__(self, q_matrix):
        self.Q = q_matrix

    def solve(self, offset=0.0, n_iters=5000, temp_start=1.0, temp_end=1e-3, seed=None):
        """
        Very simple simulated annealing for small QUBOs.
        Returns best binary vector found and energy.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        m = self.Q.shape[0]
        # random initial solution
        x = [random.choice([0,1]) for _ in range(m)]
        best = list(x)
        best_e = self.qubo_energy(x, offset)
        cur_e = best_e
        for t in range(1, n_iters+1):
            temp = temp_start * ( (temp_end/temp_start) ** (t / n_iters) )
            # flip random bit
            i = random.randrange(m)
            x_new = list(x)
            x_new[i] = 1 - x_new[i]
            e_new = self.qubo_energy(x_new, offset)
            delta = e_new - cur_e
            if delta < 0 or random.random() < math.exp(-delta / temp):
                x = x_new
                cur_e = e_new
                if cur_e < best_e:
                    best_e = cur_e
                    best = list(x)
        print(f"Best energy: {best_e}, best selection: {best}")
        return best, best_e
    
    def qubo_energy(self, x, offset=0.0):
        xv = np.array(x, dtype=float)
        return float(xv @ self.Q @ xv + offset)