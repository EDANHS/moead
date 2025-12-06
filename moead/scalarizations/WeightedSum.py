import numpy as np

from moead.scalarizations import Scalarization
from moead.solutions import Solution


class WeightedSum(Scalarization):
    """
    Implementación concreta de Suma Ponderada.
    g(x | λ) = sum_j { λ_j * f_j(x) }
    
    NOTA: Falla en frentes cóncavos (ej. ZDT2).
    """
    def execute(self, 
                solution: Solution, 
                lambda_vec: np.ndarray, 
                z_star: np.ndarray) -> float:
        
        # z_star es ignorado, pero debe estar en la firma
        # para cumplir con la interfaz.
        return np.sum(lambda_vec * solution.objectives)
    