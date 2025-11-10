from abc import ABC, abstractmethod
import numpy as np
from solutions import Solution

# (Aquí irían las otras interfaces: Problem, Solution, Crossover, Mutation)

class Scalarization(ABC):
    """
    Define la interfaz para una función de escalarización (fitness).
    """
    @abstractmethod
    def execute(self, 
                solution: Solution, 
                lambda_vec: np.ndarray, 
                z_star: np.ndarray) -> float:
        """
        Calcula el valor de fitness (un solo escalar) para una solución
        dado un vector de peso y el punto ideal.
        """
        raise NotImplementedError