from abc import ABC, abstractmethod

from moead.solutions import Solution

class Crossover(ABC):
    """Define la interfaz para un operador de Crossover."""
    @abstractmethod
    def execute(self, parent1: Solution, parent2: Solution, bounds: list) -> Solution:
        """Toma dos padres y devuelve un nuevo hijo (Solución)."""
        raise NotImplementedError