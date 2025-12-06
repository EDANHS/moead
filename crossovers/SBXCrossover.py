import numpy as np

from crossovers import Crossover
from solutions import Solution, ZDTSolution


class SBXCrossover(Crossover):
    """
    Implementación concreta de Simulated Binary Crossover (SBX).
    
    Parámetros:
    - eta: El índice de distribución del cruce (controla qué tan
           similares son los hijos a los padres). Un valor más alto
           genera hijos más parecidos.
    - prob_cross: Probabilidad de que ocurra el cruce.
    """
    def __init__(self, eta: float = 15.0, prob_cross: float = 0.9):
        self.eta = eta
        self.prob_cross = prob_cross

    def execute(self, parent1: Solution, parent2: Solution, bounds: list) -> Solution:
        """
        Toma dos padres y devuelve una nueva Solución hija.
        """
        p1_vars = parent1.variables
        p2_vars = parent2.variables
        
        # Si no se aplica el cruce, devolver un clon del primer padre
        if np.random.rand() > self.prob_cross:
            return ZDTSolution(np.copy(p1_vars), parent1.objectives.shape[0], parent1.constraints.shape[0])

        # Lógica matemática de SBX
        rand = np.random.random(p1_vars.shape)
        beta = np.empty(p1_vars.shape)
        
        alpha = 2.0 - (1.0 / (1.0 + (2.0 * rand)**(1.0 / (self.eta + 1.0))))
        beta[rand <= 0.5] = alpha[rand <= 0.5]**- (self.eta + 1.0)
        
        alpha = 2.0 - (1.0 / (1.0 + (2.0 * (1.0 - rand))**(1.0 / (self.eta + 1.0))))
        beta[rand > 0.5] = alpha[rand > 0.5]**(self.eta + 1.0)

        # Generar el hijo (solo usamos uno en MOEA/D)
        c1 = 0.5 * ((p1_vars + p2_vars) - beta * (p2_vars - p1_vars))
        
        # Aplicar límites (clipping)
        min_b = np.array([b[0] for b in bounds])
        max_b = np.array([b[1] for b in bounds])
        c1 = np.clip(c1, min_b, max_b)
        
        # Devolver la nueva solución hija
        return ZDTSolution(c1, parent1.objectives.shape[0], parent1.constraints.shape[0])