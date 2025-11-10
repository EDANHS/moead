import numpy as np

from solutions import Solution

class ZDTSolution(Solution):
    """
    Implementación concreta de Solución.
    Almacena variables (array de numpy) y objetivos (array de numpy).
    """
    
    def __init__(self, variables: np.ndarray, n_objectives: int):
        self._variables = variables
        self._objectives = np.full(n_objectives, np.inf)

    @property
    def objectives(self) -> np.ndarray: 
        return self._objectives
    
    @objectives.setter
    def objectives(self, values: np.ndarray): 
        self._objectives = values

    @property
    def variables(self) -> np.ndarray: 
        return self._variables
        
    @variables.setter
    def variables(self, values: np.ndarray): 
        self._variables = values