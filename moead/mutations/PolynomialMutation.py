from . import Mutation
import numpy as np
from moead.solutions import Solution



class PolynomialMutation(Mutation):
    """
    Implementación concreta de Mutación Polinómica.
    
    Parámetros:
    - eta: El índice de distribución de la mutación.
    - prob_mut: Probabilidad de mutar cada variable.
                Si es 'None', se usará 1 / n_variables.
    """
    def __init__(self, eta: float = 20.0, prob_mut: float = None):
        self.eta = eta
        self.prob_mut = prob_mut

    def execute(self, solution: Solution, bounds: list) -> Solution:
        """
        Modifica la solución 'en el sitio' (in-place) y la devuelve.
        """
        x = solution.variables
        n_vars = len(x)
        
        # Definir la probabilidad de mutación
        prob_mut = self.prob_mut if self.prob_mut is not None else (1.0 / n_vars)
        
        # Preparar los límites
        min_b = np.array([b[0] for b in bounds])
        max_b = np.array([b[1] for b in bounds])
        
        x_mut = np.copy(x)
        
        # Iterar sobre cada variable
        for i in range(n_vars):
            if np.random.rand() < prob_mut:
                # Omitir si la variable tiene un límite fijo
                if min_b[i] == max_b[i]:
                    continue 
                    
                # Lógica matemática de Mutación Polinómica
                delta1 = (x[i] - min_b[i]) / (max_b[i] - min_b[i])
                delta2 = (max_b[i] - x[i]) / (max_b[i] - min_b[i])
                
                rand = np.random.random()
                mut_pow = 1.0 / (self.eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy**(self.eta + 1.0))
                    delta_q = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy**(self.eta + 1.0))
                    delta_q = 1.0 - (val**mut_pow)

                # Aplicar la mutación
                x_mut[i] = x[i] + delta_q * (max_b[i] - min_b[i])
        
        # Aplicar límites (clipping)
        x_mut = np.clip(x_mut, min_b, max_b)
        
        # Actualizar la solución y devolverla
        solution.variables = x_mut
        return solution