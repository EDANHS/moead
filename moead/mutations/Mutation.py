from abc import ABC, abstractmethod

from moead.solutions import Solution

class Mutation(ABC):
    """Define la interfaz para un operador de Mutación."""
    @abstractmethod
    def execute(self, solution: Solution, bounds: list) -> Solution:
        """Modifica una solución y la devuelve."""
        raise NotImplementedError