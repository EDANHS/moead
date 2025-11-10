import numpy as np

from scalarizations import Scalarization
from solutions import Solution

class PBI(Scalarization):
    """
    Implementación de Penalty-based Boundary Intersection (PBI).
    g(x | λ, z*) = d1 + θ * d2
    
    'theta' (θ) es el parámetro de penalización.
    """
    def __init__(self, theta: float = 5.0):
        """
        Inicializa PBI con un parámetro de penalización theta.
        Valores comunes de theta están entre 1.0 y 10.0.
        """
        self.theta = theta

    def execute(self, 
                solution: Solution, 
                lambda_vec: np.ndarray, 
                z_star: np.ndarray) -> float:
        
        # Vector de la solución al punto ideal
        diff = solution.objectives - z_star
        
        # Norma L2 del vector de peso (denominador)
        # Añadimos un épsilon (1e-10) por si un vector lambda
        # fuera [0, 0, ...] para evitar división por cero.
        lambda_norm = np.linalg.norm(lambda_vec)
        if lambda_norm < 1e-10:
            lambda_norm = 1e-10 # Evitar división por cero
        
        # d1: Distancia de convergencia (proyección)
        # Es el producto punto de 'diff' y el vector lambda unitario
        d1 = np.dot(diff, lambda_vec) / lambda_norm
        
        # d2: Distancia de penalización (perpendicular)
        # Es la norma del vector 'diff' menos su proyección
        projection_vec = (d1 / lambda_norm) * lambda_vec
        d2 = np.linalg.norm(diff - projection_vec)
        
        return d1 + self.theta * d2