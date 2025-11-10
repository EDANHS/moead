from abc import ABC, abstractmethod
import numpy as np

from problems.Problem import Problem
from solutions import Solution
# (Asume que Solution, Problem, etc. están definidos)

class EvolutionaryOperator(ABC):
    """
    Interfaz para una "Estrategia de Reproducción".
    Encapsula CÓMO se genera un nuevo hijo.
    """
    
    @abstractmethod
    def execute(self, 
                i: int,           
                population: list[Solution], 
                neighborhoods: np.ndarray, 
                problem: Problem,
                **kwargs) -> Solution:
        """
        Genera un nuevo hijo para el subproblema 'i'.
        """
        raise NotImplementedError