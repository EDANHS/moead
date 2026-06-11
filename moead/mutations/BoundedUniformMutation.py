import numpy as np
from . import Mutation
from moead.solutions.Solution import Solution

class BoundedUniformMutation(Mutation):
    """
    Implementación de Mutación Uniforme Acotada para espacios de búsqueda discretizados.
    Preserva la viabilidad del genotipo sustituyendo genes de forma estocástica
    por valores muestreados uniformemente dentro de sus límites dimensionales.
    """
    def __init__(self, prob_mut: float = 0.15):
        """
        Ajusta la tasa de mutación por gen. 
        Un valor estándar recomendado para NAS es 1/n_variables.
        """
        self.prob_mut = prob_mut

    def execute(self, solution: Solution, bounds: list) -> Solution:
        # Extraer una copia profunda del vector de variables genéticas
        child_vars = np.copy(solution.variables)
        
        # Iterar sobre cada dimensión del espacio de búsqueda y sus límites
        for idx, (min_b, max_b) in enumerate(bounds):
            if np.random.rand() < self.prob_mut:
                # Muestreo uniforme puro dentro del hipercubo de la variable
                child_vars[idx] = np.random.uniform(min_b, max_b)
                
        # Asignar el nuevo vector modificado a la solución existente
        solution.variables = child_vars
        return solution