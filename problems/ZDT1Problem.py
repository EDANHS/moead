import numpy as np

from problems import Problem
from solutions import Solution, ZDTSolution

class ZDT1Problem(Problem):
    """
    Implementación concreta del problema ZDT1.
    
    Responsabilidades:
    - Definir el número de objetivos (M=2).
    - Definir los límites (bounds) de las variables.
    - Crear una solución aleatoria inicial.
    - Evaluar una solución dada (calcular f1 y f2).
    """
    
    def __init__(self, n_vars: int = 30):
        self._n_objectives = 2
        self.n_vars = n_vars
        self._bounds = [(0.0, 1.0)] * self.n_vars

    @property
    def n_objectives(self) -> int:
        return self._n_objectives
        
    @property
    def bounds(self) -> list[tuple[float, float]]:
        return self._bounds
        
    def create_solution(self) -> Solution:
        """
        Crea una solución aleatoria con variables entre 0 y 1.
        """
        vars = np.random.rand(self.n_vars)
        return ZDTSolution(vars, self.n_objectives)
        
    def evaluate(self, solution: Solution):
        """
        Calcula f1 y f2 para la solución y actualiza sus objetivos.
        """
        x = solution.variables
        
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_vars - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        
        solution.objectives = np.array([f1, f2])