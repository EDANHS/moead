from abc import ABC, abstractmethod

from moead.solutions import Solution

class Problem(ABC):
    """
    Define la interfaz para un problema de optimización.
    Ahora incluye el número de restricciones y 'evaluate' debe calcularlas.
    """
    @property
    @abstractmethod
    def n_objectives(self) -> int:
        """El número de objetivos (M)."""
        raise NotImplementedError
        
    @property
    @abstractmethod
    def n_constraints(self) -> int:
        """El número de restricciones."""
        raise NotImplementedError
        
    @property
    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]:
        """Límites de las variables [(min1, max1), ...]."""
        raise NotImplementedError
        
    @abstractmethod
    def create_solution(self) -> Solution:
        """Crea una instancia de Solución aleatoria."""
        raise NotImplementedError
        
    @abstractmethod
    def evaluate(self, solution: Solution):
        """
        Calcula y actualiza los 'objectives' Y los 'constraints'
        de una solución dada.
        """
        raise NotImplementedError