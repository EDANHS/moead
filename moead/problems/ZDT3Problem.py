import numpy as np

from moead.problems import Problem
from moead.solutions import Solution, ZDTSolution

class ZDT3Problem(Problem):
    """
    Implementación concreta del problema ZDT3 (Frente Discontinuo).
    
    Responsabilidades:
    - Definir el número de objetivos (M=2).
    - Definir los límites (bounds) de las variables.
    - Crear una solución aleatoria inicial.
    - Evaluar una solución dada (calcular f1 y f2).
    """
    
    def __init__(self, n_vars: int = 30):
        self._n_objectives = 2
        self._n_constraints = 0
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
        
        # Fórmula de f2 específica de ZDT3
        term1 = np.sqrt(f1 / g)
        term2 = (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * (1.0 - term1 - term2)
        
        solution.objectives = np.array([f1, f2])
        solution.constraints = np.array([])