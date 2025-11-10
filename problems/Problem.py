from abc import ABC, abstractmethod

from solutions import Solution

class Problem(ABC):
    """
    Define la interfaz para un problema de optimización.
    (Nota: Ya no es responsable de Crossover o Mutación).
    """
    @property
    @abstractmethod
    def n_objectives(self) -> int:
        """El número de objetivos (M)."""
        raise NotImplementedError
        
    @property
    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]:
        """Límites de las variables [(min1, max1), (min2, max2), ...]."""
        raise NotImplementedError
        
    @abstractmethod
    def create_solution(self) -> Solution:
        """Crea una instancia de Solución aleatoria."""
        raise NotImplementedError
        
    @abstractmethod
    def evaluate(self, solution: Solution):
        """Calcula y actualiza los objetivos de una solución."""
        raise NotImplementedError