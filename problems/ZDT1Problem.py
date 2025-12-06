import numpy as np

from problems import Problem
from solutions import Solution, ZDTSolution

class ZDT1Problem(Problem):
    """Implementación de ZDT1 (sin restricciones)."""
    def __init__(self, n_vars: int = 30):
        self._n_objectives = 2
        self._n_constraints = 0 # ZDT1 no tiene restricciones
        self.n_vars = n_vars
        self._bounds = [(0.0, 1.0)] * self.n_vars

    @property
    def n_objectives(self) -> int: return self._n_objectives
    @property
    def n_constraints(self) -> int: return self._n_constraints
    @property
    def bounds(self) -> list[tuple[float, float]]: return self._bounds
        
    def create_solution(self) -> Solution:
        vars = np.random.rand(self.n_vars)
        return ZDTSolution(vars, self.n_objectives, self.n_constraints)
        
    def evaluate(self, solution: Solution):
        x = solution.variables
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_vars - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        
        solution.objectives = np.array([f1, f2])
        solution.constraints = np.array([]) # Devuelve array vacío