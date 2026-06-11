import numpy as np

from . import Crossover
from moead.solutions import Solution


class UniformCrossover(Crossover):
    """
    Uniform crossover: para cada gen, elige aleatoriamente del padre1 o padre2.
    Garantiza la preservación de bloques de construcción discretos e implementa
    clipping automático para evitar desbordamientos de límites.
    """
    def __init__(self, prob_cross: float = 0.9):
        self.prob_cross = prob_cross

    def execute(self, 
                parent1: Solution, 
                parent2: Solution, 
                bounds: list, 
                debugger=None, 
                **kwargs) -> Solution:
        
        p1 = parent1.variables
        p2 = parent2.variables

        # Telemetría opcional del inicio del cruce
        if debugger is not None:
            debugger.start_step('uniform_crossover', {
                'prob_cross': self.prob_cross,
                'parent1_variables': p1.tolist(),
                'parent2_variables': p2.tolist()
            })

        # Evaluación de la probabilidad estocástica de cruce
        if np.random.rand() > self.prob_cross:
            if debugger is not None:
                debugger.pass_step('uniform_crossover', 'No crossover applied, returning clone of parent 1')
            # Retorna un clon genético sin vaciar metadatos para simular supervivencia estática
            return type(parent1)(np.copy(p1), parent1.objectives.shape[0], parent1.constraints.shape[0])

        # Generación de la máscara de recombinación uniforme (50% de probabilidad por gen)
        mask = np.random.rand(*p1.shape) < 0.5
        child_vars = np.where(mask, p1, p2)

        # Aplicación estricta de la frontera geométrica (Clipping)
        min_b = np.array([b[0] for b in bounds])
        max_b = np.array([b[1] for b in bounds])
        child_vars = np.clip(child_vars, min_b, max_b)

        if debugger is not None:
            debugger.pass_step('uniform_crossover', 'Uniform crossover completed successfully', {
                'child_variables': child_vars.tolist(),
                'mask_applied': mask.tolist()
            })

        # Ensamblaje y retorno del nuevo fenotipo aislado
        return type(parent1)(child_vars, parent1.objectives.shape[0], parent1.constraints.shape[0])
