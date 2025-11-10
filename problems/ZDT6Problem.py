import numpy as np

from problems import Problem
from solutions import Solution, ZDTSolution

class ZDT6Problem(Problem):
    """
    Implementación concreta del problema ZDT6 (Frente Cóncavo y No Uniforme).
    
    Responsabilidades:
    - Definir el número de objetivos (M=2).
    - Definir los límites (bounds) de las variables.
    - Crear una solución aleatoria inicial.
    - Evaluar una solución dada (calcular f1 y f2).
    """
    
    def __init__(self, n_vars: int = 10):
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
        
        # --- Fórmula f1 específica de ZDT6 ---
        f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])**6)
        # --------------------------------------
        
        # --- Función g(x) específica de ZDT6 ---
        g_sum = np.sum(x[1:])
        g = 1.0 + 9.0 * (g_sum / (self.n_vars - 1))**0.25
        # ---------------------------------------
        
        # La forma de f2 es la misma que ZDT2
        f2 = g * (1.0 - (f1 / g)**2)
        
        solution.objectives = np.array([f1, f2])