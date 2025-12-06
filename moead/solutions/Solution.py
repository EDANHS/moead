from abc import ABC, abstractmethod
import numpy as np

class Solution(ABC):
    """
    Define la interfaz para una solución.
    Ahora incluye propiedades para objetivos, variables y restricciones.
    """
    @property
    @abstractmethod
    def objectives(self) -> np.ndarray:
        """Devuelve un array de numpy con los M valores de objetivo."""
        raise NotImplementedError
    
    @objectives.setter
    @abstractmethod
    def objectives(self, values: np.ndarray):
        raise NotImplementedError

    @property
    @abstractmethod
    def variables(self) -> any:
        """Devuelve las variables de decisión (ej. un array)."""
        raise NotImplementedError
        
    @variables.setter
    @abstractmethod
    def variables(self, values: any):
        raise NotImplementedError
        
    @property
    @abstractmethod
    def constraints(self) -> np.ndarray:
        """
        Devuelve un array de las violaciones de las restricciones.
        - Un valor <= 0 significa que la restricción se cumple.
        - Un valor > 0 significa que la restricción SE VIOLA.
        """
        raise NotImplementedError
    
    @constraints.setter
    @abstractmethod
    def constraints(self, values: np.ndarray):
        raise NotImplementedError