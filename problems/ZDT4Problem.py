import numpy as np

from problems import Problem
from solutions import Solution, ZDTSolution

class ZDT4Problem(Problem):
    """
    Implementación concreta del problema ZDT4 (Múltiples Óptimos Locales).
    
    ¡CUIDADO! Este problema tiene bounds diferentes y un 'g(x)' diferente.
    Se suele probar con n_vars = 10.
    """
    
    def __init__(self, n_vars: int = 10):
        self._n_objectives = 2
        self.n_vars = n_vars
        
        # --- ¡Bounds diferentes! ---
        self._bounds = [(0.0, 1.0)] + [(-5.0, 5.0)] * (self.n_vars - 1)
        # --------------------------

    @property
    def n_objectives(self) -> int:
        return self._n_objectives
        
    @property
    def bounds(self) -> list[tuple[float, float]]:
        return self._bounds
        
    def create_solution(self) -> Solution:
        """
        Crea una solución aleatoria respetando los bounds de ZDT4.
        """
        vars = np.empty(self.n_vars)
        vars[0] = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
        vars[1:] = np.random.uniform(self.bounds[1][0], self.bounds[1][1], self.n_vars - 1)
        
        return ZDTSolution(vars, self.n_objectives)
        
    def evaluate(self, solution: Solution):
        """
        Calcula f1 y f2 para la solución y actualiza sus objetivos.
        """
        x = solution.variables
        
        f1 = x[0]
        
        # --- Función g(x) específica de ZDT4 ---
        g_vars = x[1:]
        g_sum = np.sum(g_vars**2 - 10 * np.cos(4 * np.pi * g_vars))
        g = 1.0 + 10 * (self.n_vars - 1) + g_sum
        # ---------------------------------------
        
        # La forma de f2 es la misma que ZDT1
        f2 = g * (1.0 - np.sqrt(f1 / g))
        
        solution.objectives = np.array([f1, f2])