import numpy as np

from moead.scalarizations import Scalarization
from moead.solutions.Solution import Solution


class Tchebycheff(Scalarization):
    """
    Implementación concreta de Tchebycheff.
    g(x | λ, z*) = max_j { λ_j * |f_j(x) - z*_j| }
    """
    def execute(self, 
                solution: Solution, 
                lambda_vec: np.ndarray, 
                z_star: np.ndarray) -> float:
        
        obj = solution.objectives
        weighted_diff = lambda_vec * np.abs(obj - z_star)
        return np.max(weighted_diff)