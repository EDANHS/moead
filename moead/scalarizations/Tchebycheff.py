import numpy as np

from moead.scalarizations import Scalarization
from moead.solutions.Solution import Solution

class Tchebycheff(Scalarization):
    """
    Implementación concreta de Tchebycheff con corrección asintótica.
    g(x | λ, z*) = max_j { λ_j * |f_j(x) - z*_j| }
    Incorpora un factor epsilon para evitar la pérdida de gradiente 
    selectivo cuando las componentes del vector lambda son cero.
    """
    def __init__(self, epsilon: float = 1e-6):
        # Epsilon actúa como un peso mínimo residual (típicamente 1e-4 o 1e-6)
        self.epsilon = epsilon

    def execute(self, 
                solution: Solution, 
                lambda_vec: np.ndarray, 
                z_star: np.ndarray) -> float:
        
        obj = solution.objectives
        
        # 1. Estrategia de Inmunidad: Reemplazar ceros absolutos por epsilon
        # Esto mantiene la vectorización rápida de C en NumPy sin usar bucles "for"
        safe_lambda = np.where(lambda_vec == 0, self.epsilon, lambda_vec)
        
        # 2. Computar la diferencia ponderada de forma segura
        weighted_diff = safe_lambda * np.abs(obj - z_star)
        
        return float(np.max(weighted_diff))