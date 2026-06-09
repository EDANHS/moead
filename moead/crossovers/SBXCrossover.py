import numpy as np

from moead.crossovers import Crossover
from moead.solutions import Solution


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

    def execute(self, parent1: Solution, parent2: Solution, bounds: list, debugger=None) -> Solution:
        """
        Toma dos padres y devuelve una nueva Solución hija.
        """
        p1_vars = parent1.variables
        p2_vars = parent2.variables
        if debugger is not None:
            debugger.start_step('sbx_crossover', {
                'prob_cross': self.prob_cross,
                'eta': self.eta,
                'parent1_variables': p1_vars.tolist(),
                'parent2_variables': p2_vars.tolist()
            })
        
        # Si no se aplica el cruce, devolver un clon del primer padre
        if np.random.rand() > self.prob_cross:
            child = type(parent1)(np.copy(p1_vars), parent1.objectives.shape[0], parent1.constraints.shape[0])
            if debugger is not None:
                debugger.pass_step('sbx_crossover', 'No crossover applied, returning clone')
            return child

        # Lógica matemática de SBX
        rand = np.random.random(p1_vars.shape)
        beta = np.empty(p1_vars.shape)
        
        alpha = 2.0 - (1.0 / (1.0 + (2.0 * rand)**(1.0 / (self.eta + 1.0))))
        beta[rand <= 0.5] = alpha[rand <= 0.5]**- (self.eta + 1.0)
        
        alpha = 2.0 - (1.0 / (1.0 + (2.0 * (1.0 - rand))**(1.0 / (self.eta + 1.0))))
        beta[rand > 0.5] = alpha[rand > 0.5]**(self.eta + 1.0)

        # Generar el hijo (solo usamos uno en MOEA/D)
        c1 = 0.5 * ((p1_vars + p2_vars) - beta * (p2_vars - p1_vars))
        if debugger is not None:
            debugger.record_event('sbx_crossover', 'info', 'Child candidate computed', {
                'child_variables': c1.tolist()
            })

        # Validación estricta de límites: no se hace clipping suave.
        min_b = np.array([b[0] for b in bounds])
        max_b = np.array([b[1] for b in bounds])
        invalid = np.any(c1 < min_b) or np.any(c1 > max_b)

        child = type(parent1)(c1, parent1.objectives.shape[0], parent1.constraints.shape[0])
        if invalid:
            child.objectives = np.full(parent1.objectives.shape, np.inf)
            if child.constraints.shape[0] > 0:
                child.constraints = np.full(child.constraints.shape, np.inf)
            setattr(child, '_invalid_genotype', True)
            if debugger is not None:
                debugger.warning_step('sbx_crossover', 'Child violates bounds after SBX', {
                    'child_variables': c1.tolist()
                })
            return child

        if debugger is not None:
            debugger.pass_step('sbx_crossover', 'SBX crossover completed successfully', {
                'child_variables': c1.tolist()
            })
        return child