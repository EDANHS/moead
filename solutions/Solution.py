from abc import ABC, abstractmethod
import numpy as np

class Solution(ABC):
    """
    Define la interfaz para una solución.
    El solver solo necesita saber cómo acceder a sus variables y objetivos.
    """
    @property
    @abstractmethod
    def objectives(self) -> np.ndarray:
        raise NotImplementedError
    
    @objectives.setter
    @abstractmethod
    def objectives(self, values: np.ndarray):
        raise NotImplementedError

    @property
    @abstractmethod
    def variables(self) -> any:
        raise NotImplementedError
        
    @variables.setter
    @abstractmethod
    def variables(self, values: any):
        raise NotImplementedError